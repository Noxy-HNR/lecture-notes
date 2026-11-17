"""Continuously drains the microphone/system-audio buffer on a dedicated background
thread, decoupled from transcription and saving.

Why this exists: WASAPI's hardware capture buffer is small (a fraction of a second to
a couple seconds). If the app doesn't call into the recorder often enough to drain it,
the OS silently overflows the buffer and drops audio - not delays it, drops it. In the
single-threaded design this app started with, ANY slow step in the main loop (a stuck
Claude API call, and especially the multi-minute speaker-diarization pass that used to
run every autosave) blocked the next read for as long as that step took, silently
losing whatever was said during the gap. This was confirmed live: diarization blocking
the loop caused a real ~2 minute gap of lost lecture audio.

CaptureThread fixes this categorically rather than case-by-case: it does nothing but
read small blocks from the recorder in a tight loop and push them onto a thread-safe
queue.Queue, as fast as the hardware provides them. The queue can hold many minutes of
audio in memory, so however slow downstream processing (transcription, saving,
diarization, a network call) gets, it only adds latency to when segments show up in the
live view - it can never again cause audio to be silently dropped, because nothing
downstream is on the same thread as the thing draining the hardware buffer.
"""
import collections
import functools
import queue
import threading
import time

import numpy as np

import audio

READ_SECONDS = 0.5  # small blocks so the OS buffer gets drained frequently
RECORDER_RETRY_SECONDS = 2
STARTUP_SAMPLE_SECONDS = 3.0

# What on_block returns to say "I'm keeping this block, don't queue it for transcription"
# (see Preroll: audio captured before a session exists to write it to).
HOLD = object()

# The most audio kept from before Record was chosen. The app can sit at its menu for a long
# time; a lecture that started while it did is worth keeping, an hour of hallway isn't.
PREROLL_MAX_SECONDS = 600.0

# Pause-aware chunking: end a chunk on a natural speech pause rather than a fixed timer,
# so boundaries don't land mid-word. Implemented, measured, and left OFF by default -
# it made transcription measurably WORSE on this app's real workload.
#
#   Measured with the since-removed Whisper model (tools/vad_ab_test.py, also removed),
#   5 min per clip, WER against a single-pass reference
#   (no chunk boundaries at all, so it isolates boundary damage):
#
#     PSYC 1300 (lecturer A):  fixed 15.30%  ->  pause-aware 16.67%   (+1.37 worse)
#     BIOL 1440 (lecturer B):  fixed 26.69%  ->  pause-aware 29.51%   (+2.82 worse)
#
# Consistent direction across two speakers and rooms, and slower besides (more chunks =
# more per-chunk overhead). The likely reason is that an energy-based detector doesn't
# actually find sentence boundaries - it finds any brief dip, including breaths and
# mid-clause gaps - so it cuts mid-thought more often than a fixed timer does, while
# also handing the model shorter chunks (mean 12-13s vs 15s) with less context to work
# with. Cohere now cuts its own 120s windows at quiet points (see transcribe.py).
#
# Kept (rather than deleted) so the finding stays reproducible - flip pause_aware=True in
# collect_chunk() to re-test if the thresholds or the model ever change. The tail-silence test is RELATIVE to surrounding loudness, not
# an absolute level: audio.is_silent()'s threshold is tuned to detect a dead/muted mic
# (RMS < 0.002), and a real lecture hall's noise floor measured ~0.013, so an absolute
# test would essentially never fire.
# Thresholds below are measured, not guessed. Against two real lectures (different
# rooms, speakers and mics), the tail-vs-context RMS ratio has a median around 0.8 and
# a 1st percentile around 0.2, and a 0.25 cutoff fires roughly once every 4-14s
# depending on how much the lecturer pauses - often enough to land most chunk boundaries
# on a real pause, rare enough that it doesn't shred speech into fragments.
PAUSE_RATIO = 0.25
PAUSE_TAIL_SECONDS = 0.25    # the candidate "quiet" window being tested
PAUSE_CONTEXT_SECONDS = 6.0  # what it's compared against. MUST be much longer than the
                              # tail: an earlier version collected just enough audio to
                              # cover the tail, which made the comparison window and the
                              # tail the same samples, reducing the test to
                              # rms < 0.25*rms - never true, so pause detection silently
                              # never fired at all. Caught by the A/B harness reporting
                              # byte-identical chunking for both strategies.
MIN_CHUNK_SECONDS = 8.0      # never cut this early even on a pause - accuracy
                              # degrades on very short fragments with little context


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def ends_on_pause(blocks: list[np.ndarray], tail_frames: int,
                   context_frames: int | None = None) -> bool:
    """True if the last `tail_frames` samples across `blocks` are quiet relative to the
    preceding `context_frames` - i.e. the speaker has paused and this is a sensible place
    to end a chunk. Module-level (not a method) so the offline A/B harness exercises the
    exact same boundary logic the live capture loop uses, rather than a reimplementation.

    The comparison is relative because a lecture hall's noise floor (HVAC, shuffling,
    fans) sits well above audio.is_silent()'s absolute threshold - measured RMS during
    real pauses was ~0.013, against a 0.002 "is the mic dead" threshold. Only the ratio
    between a quiet moment and surrounding speech reliably identifies a pause."""
    if tail_frames <= 0 or not blocks:
        return False
    if context_frames is None:
        context_frames = int(PAUSE_CONTEXT_SECONDS * audio.SAMPLE_RATE)
    if context_frames <= tail_frames:
        return False  # a tail compared against itself can never read as quieter

    collected, have = [], 0
    for block in reversed(blocks):
        collected.append(block)
        have += len(block)
        if have >= context_frames:
            break
    recent = np.concatenate(list(reversed(collected)))
    # Not enough history yet to judge what "normal" loudness is for this stretch.
    if len(recent) < context_frames:
        return False

    context = recent[-context_frames:]
    tail = recent[-tail_frames:]
    context_rms = _rms(context)
    if context_rms <= 0:
        return True  # nothing but digital silence - fine to cut anywhere
    return _rms(tail) < max(audio.SILENCE_RMS_THRESHOLD, PAUSE_RATIO * context_rms)


class CaptureThread:
    def __init__(self, source: str, on_block=None):
        self.source = source
        self.on_block = on_block
        self.fatal_error = None
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.error_queue: "queue.Queue[Exception]" = queue.Queue()
        self._stop_event = threading.Event()
        self._abandoned = False
        self.started_at = None
        self.last_audio_at = None  # wall-clock time the device last delivered a block
        self.discontinuities = 0  # WASAPI reported gaps; not a duration estimate
        self._startup_ready = threading.Event()
        self._startup_lock = threading.Lock()
        self._startup_parts = []
        self._startup_frames = 0
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        # Blocks already pulled off audio_queue by collect_chunk() but not yet returned
        # (because it's still accumulating toward target_seconds). Kept on self, not a
        # local variable, so that if Ctrl+C raises KeyboardInterrupt while collect_chunk
        # is blocked waiting for the next block, whatever it already pulled isn't lost -
        # drain_available() reunites it with the rest of the queue afterward.
        self._accumulating: list[np.ndarray] = []

    def start(self):
        self.started_at = time.time()
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def startup_sample(self, seconds=STARTUP_SAMPLE_SECONDS, timeout=10.0):
        """Observe persisted first audio without consuming the transcription queue."""
        if not self._startup_ready.wait(timeout):
            raise RuntimeError(self.fatal_error or 'audio device did not deliver its startup sample')
        with self._startup_lock:
            return np.concatenate(self._startup_parts)[:int(seconds * audio.SAMPLE_RATE)]

    def reset_startup_sample(self, recent=None):
        """Takes the startup sample again from the audio that arrives next - or, given at least
        STARTUP_SAMPLE_SECONDS of `recent` audio, uses the end of that, so a capture that has
        been running since the app opened doesn't make the audio check wait another 3 s."""
        target = int(STARTUP_SAMPLE_SECONDS * audio.SAMPLE_RATE)
        with self._startup_lock:
            self._startup_parts, self._startup_frames = [], 0
            self._startup_ready.clear()
            if recent is not None and len(recent) >= target:
                self._startup_parts = [np.asarray(recent[-target:], dtype=np.float32).copy()]
                self._startup_frames = target
                self._startup_ready.set()

    def _observe_startup(self, block):
        target = int(STARTUP_SAMPLE_SECONDS * audio.SAMPLE_RATE)
        with self._startup_lock:
            if self._startup_frames < target:
                part = block[:target - self._startup_frames].copy()
                self._startup_parts.append(part)
                self._startup_frames += len(part)
                if self._startup_frames >= target:
                    self._startup_ready.set()

    def seconds_since_audio(self, now: float | None = None) -> float:
        """Wall-clock seconds since the device last delivered audio (or since start)."""
        now = time.time() if now is None else now
        last = self.last_audio_at or self.started_at or now
        return max(0.0, now - last)

    def abandon(self) -> np.ndarray | None:
        """Gives up on a capture whose device stopped responding. After a laptop sleeps, the
        Windows audio call inside the capture thread can block forever without raising, so
        the thread can't be stopped or joined - it's told to exit without writing anything
        more if the driver ever returns, and the audio it had already delivered is handed
        back so none of it is lost."""
        self._abandoned = True
        self._stop_event.set()
        return self.drain_available()

    def collect_chunk(self, target_seconds: float, poll_seconds: float = 0.5,
                       interrupt_event: "threading.Event | None" = None,
                       pause_aware: bool = False,
                       max_wait_seconds: float | None = None) -> np.ndarray | None:
        """Blocks (checking `self._stop_event` periodically so it stays interruptible)
        until `target_seconds` worth of audio has been pulled from the queue, or stop
        was requested - in which case whatever partial audio is already queued (if any)
        is returned instead, so the tail end of a session isn't silently discarded.
        Returns None only if there's truly nothing available.

        `interrupt_event`, if given, ends collection early (returning whatever's
        accumulated so far, however little) without stopping capture itself - used for
        a manual "save now" request, so it doesn't have to wait for a full chunk to
        finish collecting before the save can happen. Caller is responsible for
        clearing the event afterward; this method only reads it.

        `pause_aware` ends the chunk early once at least MIN_CHUNK_SECONDS has been
        collected AND the audio is currently in a pause, making `target_seconds` an
        upper bound rather than an exact size. OFF by default: measured worse for
        accuracy on real lectures - see the constants above for the numbers.

        If a KeyboardInterrupt fires while this is blocked waiting for the next block,
        it propagates normally (the caller's Ctrl+C handling still works) - but whatever
        had already been pulled off the queue stays in self._accumulating rather than
        being lost in a local variable, so a follow-up drain_available() call recovers it.

        `max_wait_seconds`, if given, returns early (with whatever has accumulated, possibly
        None) once no new audio has arrived for that long, so a device that stopped
        delivering can't hold the caller here forever - the recording loop relies on this to
        notice a frozen capture and reconnect."""
        target_frames = int(target_seconds * audio.SAMPLE_RATE)
        min_frames = int(min(MIN_CHUNK_SECONDS, target_seconds) * audio.SAMPLE_RATE)
        tail_frames = int(PAUSE_TAIL_SECONDS * audio.SAMPLE_RATE)
        total_frames = sum(len(b) for b in self._accumulating)
        last_progress = time.monotonic()
        while total_frames < target_frames:
            if interrupt_event is not None and interrupt_event.is_set():
                break
            try:
                block = self.audio_queue.get(timeout=poll_seconds)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                if max_wait_seconds is not None and time.monotonic() - last_progress >= max_wait_seconds:
                    break
                continue
            last_progress = time.monotonic()
            self._accumulating.append(block)
            total_frames += len(block)
            if (pause_aware and total_frames >= min_frames
                    and ends_on_pause(self._accumulating, tail_frames)):
                break  # natural sentence break - better boundary than the hard cap
        if not self._accumulating:
            return None
        result = np.concatenate(self._accumulating)
        self._accumulating = []
        return result

    def drain_available(self) -> np.ndarray | None:
        """Non-blocking: returns whatever's currently queued or was mid-accumulation
        (no waiting for a target duration), or None if there's nothing at all. Used to
        pick up any last audio after stop() so the final chunk isn't lost."""
        collected = list(self._accumulating)
        self._accumulating = []
        while True:
            try:
                collected.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        if not collected:
            return None
        return np.concatenate(collected)

    def poll_errors(self) -> list[Exception]:
        """Non-blocking: returns (and clears) any capture errors logged since the last
        call. The capture thread already retries internally - this is for the caller to
        surface what happened, not to react to it."""
        errors = []
        while True:
            try:
                errors.append(self.error_queue.get_nowait())
            except queue.Empty:
                break
        return errors

    def _run(self):
        with audio.count_discontinuities(self._count_discontinuity):
            self._capture_loop()

    def _count_discontinuity(self):
        self.discontinuities += 1

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                recorder = audio.get_recorder(self.source)
            except Exception as e:
                self.error_queue.put(e)
                time.sleep(RECORDER_RETRY_SECONDS)
                continue

            try:
                with recorder:
                    while not self._stop_event.is_set():
                        block = audio.record_chunk(recorder, READ_SECONDS)
                        if self._abandoned:
                            return  # replaced while blocked in the driver: its successor owns the WAV now
                        self.last_audio_at = time.time()
                        queued = block
                        if self.on_block is not None:
                            try:
                                persisted = self.on_block(block)  # persist before any inference queue
                                if persisted is not None:
                                    queued = persisted
                            except Exception as error:
                                self.fatal_error = error
                                self.error_queue.put(error)
                                self._stop_event.set()
                                return
                        self._observe_startup(block)
                        if queued is not HOLD:
                            self.audio_queue.put(queued)
            except Exception as e:
                # Device dropped out (sleep/resume hiccup, USB mic unplugged, etc.) -
                # log it and loop back around to reopen a fresh recorder rather than
                # letting the capture thread die silently.
                self.error_queue.put(e)
                time.sleep(RECORDER_RETRY_SECONDS)


class Preroll:
    """Records from the moment the app opens, before anyone has said what to do with it.

    The recorder asks what to do, then which audio source, and only then had a session to
    write audio into - so whatever was said while those prompts were on screen was lost. A
    Preroll opens the capture for each possible source straight away and holds what arrives
    in memory (64 KB a second per source, the newest PREROLL_MAX_SECONDS at most). Then:

      adopt(source, persist)  Record was chosen with this source. Everything held for it is
                              written through `persist` (the session's durable WAV) and
                              queued for transcription, in order, and the same capture
                              keeps running straight into the session - no gap, no second
                              device open. The other source is dropped.
      discard()               Something other than recording was chosen. Every capture is
                              stopped and the audio is dropped; none of it was written
                              anywhere.

    The lock makes the hand-over exact: a block is either held (and flushed by adopt) or
    persisted directly, never both and never out of order."""

    def __init__(self, sources, make_capture, max_seconds: float = PREROLL_MAX_SECONDS):
        self._lock = threading.Lock()
        self.started = time.monotonic()
        self._max_frames = int(max_seconds * audio.SAMPLE_RATE)
        self._held = {source: collections.deque() for source in sources}
        self._frames = {source: 0 for source in sources}
        self._persist = None
        self._adopted = None
        self.captures = {}
        for source in sources:
            self.captures[source] = make_capture(source, functools.partial(self._on_block, source))
        for capture in self.captures.values():
            capture.start()

    def _on_block(self, source, block):
        with self._lock:
            if source == self._adopted:
                return self._persist(block)
            held = self._held.get(source)
            if held is not None:
                held.append(block)
                self._frames[source] += len(block)
                while len(held) > 1 and self._frames[source] - len(held[0]) >= self._max_frames:
                    self._frames[source] -= len(held.popleft())
            return HOLD

    def has(self, source) -> bool:
        return source in self.captures and self._adopted is None

    def held_seconds(self, source) -> float:
        with self._lock:
            return self._frames.get(source, 0) / audio.SAMPLE_RATE

    def heard(self, source, seconds: float = STARTUP_SAMPLE_SECONDS, minimum: float = 1.0):
        """Whether the newest `seconds` held for this source contain any sound: True, False,
        or None when under `minimum` seconds have arrived (too little to say). Judged in
        half-second windows like the startup audio check, so one spoken word counts."""
        with self._lock:
            tail, have = [], 0
            for block in reversed(self._held.get(source, ())):
                tail.append(block)
                have += len(block)
                if have >= int(seconds * audio.SAMPLE_RATE):
                    break
        if have < int(minimum * audio.SAMPLE_RATE):
            return None
        samples = np.concatenate(list(reversed(tail)))
        window = int(0.5 * audio.SAMPLE_RATE)
        return any(not audio.is_silent(samples[i:i + window]) for i in range(0, len(samples), window))

    def adopt(self, source, persist):
        """Returns (the running capture, seconds of audio kept from before now, the last
        STARTUP_SAMPLE_SECONDS of that audio or None if there was less)."""
        capture = self.captures[source]
        with self._lock:
            held = self._held.pop(source)
            frames = self._frames.pop(source)
            recent = None
            if frames >= int(STARTUP_SAMPLE_SECONDS * audio.SAMPLE_RATE):
                tail, have = [], 0
                for block in reversed(held):
                    tail.append(block)
                    have += len(block)
                    if have >= int(STARTUP_SAMPLE_SECONDS * audio.SAMPLE_RATE):
                        break
                recent = np.concatenate(list(reversed(tail)))
            for block in held:
                capture.audio_queue.put(persist(block))
            self._persist, self._adopted = persist, source
            capture.on_block = persist  # later blocks skip this class entirely
        self.discard()
        return capture, frames / audio.SAMPLE_RATE, recent

    def discard(self, everything: bool = False):
        """Stops every capture that wasn't adopted (all of them with `everything`) and drops
        what they held. abandon() rather than stop(): a loopback device with nothing playing
        can sit inside the driver, and nobody should wait on it to choose a menu option."""
        with self._lock:
            self._held.clear()
            self._frames.clear()
            doomed = [c for s, c in self.captures.items() if everything or s != self._adopted]
            self.captures = {s: c for s, c in self.captures.items() if not everything and s == self._adopted}
        for capture in doomed:
            try:
                capture.abandon()
            except Exception:
                pass
