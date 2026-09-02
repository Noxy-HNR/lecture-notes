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


class CaptureThread:
    def __init__(self, source: str):
        self.source = source
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
                       interrupt_event: "threading.Event | None" = None) -> np.ndarray | None:
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

        If a KeyboardInterrupt fires while this is blocked waiting for the next block,
        it propagates normally (the caller's Ctrl+C handling still works) - but whatever
        had already been pulled off the queue stays in self._accumulating rather than
        being lost in a local variable, so a follow-up drain_available() call recovers it."""
        target_frames = int(target_seconds * audio.SAMPLE_RATE)
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
                        self.audio_queue.put(block)
            except Exception as e:
                # Device dropped out (sleep/resume hiccup, USB mic unplugged, etc.) -
                # log it and loop back around to reopen a fresh recorder rather than
                # letting the capture thread die silently.
                self.error_queue.put(e)
                time.sleep(RECORDER_RETRY_SECONDS)
