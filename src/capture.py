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
import queue
import threading
import time

import numpy as np

import audio

READ_SECONDS = 0.5  # small blocks so the OS buffer gets drained frequently
RECORDER_RETRY_SECONDS = 2

# Pause-aware chunking: end a chunk on a natural speech pause rather than a fixed timer,
# so boundaries don't land mid-word. Implemented, measured, and left OFF by default -
# it made transcription measurably WORSE on this app's real workload.
#
#   tools/vad_ab_test.py, 5 min per clip, WER against a single-pass reference
#   (no chunk boundaries at all, so it isolates boundary damage):
#
#     PSYC 1300 (lecturer A):  fixed 15.30%  ->  pause-aware 16.67%   (+1.37 worse)
#     BIOL 1440 (lecturer B):  fixed 26.69%  ->  pause-aware 29.51%   (+2.82 worse)
#
# Consistent direction across two speakers and rooms, and slower besides (more chunks =
# more per-chunk overhead). The likely reason is that an energy-based detector doesn't
# actually find sentence boundaries - it finds any brief dip, including breaths and
# mid-clause gaps - so it cuts mid-thought more often than a fixed timer does, while
# also handing Whisper shorter chunks (mean 12-13s vs 15s) with less context to work
# with. Whisper already gets the relevant protection elsewhere: vad_filter=True strips
# silence inside each chunk, and RollingTranscriber's overlap window plus rolling
# context prompt cover words split across a boundary.
#
# Kept (rather than deleted) so the finding stays reproducible and the harness keeps
# working - flip pause_aware=True in collect_chunk() to re-test if the thresholds or
# the model ever change. The tail-silence test is RELATIVE to surrounding loudness, not
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
                              # never fired at all. Caught by tools/vad_ab_test.py
                              # reporting byte-identical chunking for both strategies.
MIN_CHUNK_SECONDS = 8.0      # never cut this early even on a pause - Whisper accuracy
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
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        # Blocks already pulled off audio_queue by collect_chunk() but not yet returned
        # (because it's still accumulating toward target_seconds). Kept on self, not a
        # local variable, so that if Ctrl+C raises KeyboardInterrupt while collect_chunk
        # is blocked waiting for the next block, whatever it already pulled isn't lost -
        # drain_available() reunites it with the rest of the queue afterward.
        self._accumulating: list[np.ndarray] = []

    def start(self):
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def collect_chunk(self, target_seconds: float, poll_seconds: float = 0.5,
                       interrupt_event: "threading.Event | None" = None,
                       pause_aware: bool = False) -> np.ndarray | None:
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
        being lost in a local variable, so a follow-up drain_available() call recovers it."""
        target_frames = int(target_seconds * audio.SAMPLE_RATE)
        min_frames = int(min(MIN_CHUNK_SECONDS, target_seconds) * audio.SAMPLE_RATE)
        tail_frames = int(PAUSE_TAIL_SECONDS * audio.SAMPLE_RATE)
        total_frames = sum(len(b) for b in self._accumulating)
        while total_frames < target_frames:
            if interrupt_event is not None and interrupt_event.is_set():
                break
            try:
                block = self.audio_queue.get(timeout=poll_seconds)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
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
                        if self.on_block is not None:
                            try:
                                self.on_block(block)  # persist before any inference queue
                            except Exception as error:
                                self.fatal_error = error
                                self.error_queue.put(error)
                                self._stop_event.set()
                                return
                        self.audio_queue.put(block)
            except Exception as e:
                # Device dropped out (sleep/resume hiccup, USB mic unplugged, etc.) -
                # log it and loop back around to reopen a fresh recorder rather than
                # letting the capture thread die silently.
                self.error_queue.put(e)
                time.sleep(RECORDER_RETRY_SECONDS)
