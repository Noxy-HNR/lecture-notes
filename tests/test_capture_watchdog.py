"""Audio capture that freezes after a laptop sleeps (HLTH 1320, 2026-09-16): the Windows
audio call never returned, so the recorder ran for hours capturing nothing, silently.
No real audio device is used here."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import audio
import capture as capture_module
import main

SR = audio.SAMPLE_RATE


class FakeCapture:
    def __init__(self, queued_seconds=0.0):
        self.started = False
        self.abandoned = False
        self.last_audio_at = None
        self.stalled = 0.0
        self.queued = np.full(int(queued_seconds * SR), 0.1, dtype=np.float32) if queued_seconds else None

    def start(self):
        self.started = True

    def seconds_since_audio(self, now=None):
        return self.stalled

    def abandon(self):
        self.abandoned = True
        return self.queued


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def make_watchdog(first):
    clock, made = Clock(), [first]
    watchdog = main.CaptureWatchdog(lambda: made[-1] if len(made) == 1 and not made[0].started
                                    else made.append(FakeCapture()) or made[-1], clock=clock)
    return watchdog, clock, made


class TestCaptureWatchdog:
    def test_flowing_audio_needs_nothing(self):
        watchdog, clock, made = make_watchdog(FakeCapture())
        clock.now += 5
        assert watchdog.check() == ([], None) and watchdog.capture is made[0]

    def test_frozen_capture_is_replaced_and_its_audio_kept(self):
        frozen = FakeCapture(queued_seconds=1.5)
        watchdog, clock, made = make_watchdog(frozen)
        frozen.stalled, clock.now = 12.0, clock.now + 3
        messages, recovered = watchdog.check()
        assert [level for level, _ in messages] == ["error"] and "Reconnecting" in messages[0][1]
        assert frozen.abandoned and len(recovered) == int(1.5 * SR)
        assert watchdog.capture is made[1] and made[1].started and watchdog.restarts == 1

    def test_reconnect_attempts_are_spaced_out_and_recovery_is_announced(self):
        frozen = FakeCapture()
        watchdog, clock, made = make_watchdog(frozen)
        frozen.stalled, clock.now = 12.0, clock.now + 3
        watchdog.check()
        made[1].stalled, clock.now = 12.0, clock.now + 3  # replacement silent too
        assert watchdog.check() == ([], None) and watchdog.restarts == 1  # too soon to retry
        clock.now += 30
        messages, _ = watchdog.check()
        assert watchdog.restarts == 2 and "Reconnecting" in messages[0][1]
        made[2].stalled, made[2].last_audio_at, clock.now = 0.5, clock.now, clock.now + 3
        messages, _ = watchdog.check()
        assert [level for level, _ in messages] == ["success"] and "coming in again" in messages[0][1]

    def test_sleep_gap_is_reported_with_times(self):
        watchdog, clock, _ = make_watchdog(FakeCapture())
        clock.now += 3 * 3600
        messages, _ = watchdog.check()
        assert messages[0][0] == "warning" and "asleep or unresponsive" in messages[0][1]
        assert "about 180 min" in messages[0][1]


class TestCaptureThread:
    def test_collection_stops_waiting_when_no_audio_arrives(self):
        thread = capture_module.CaptureThread("mic")  # never started: no audio will ever come
        started = time.monotonic()
        assert thread.collect_chunk(5.0, poll_seconds=0.05, max_wait_seconds=0.3) is None
        assert time.monotonic() - started < 2.0

    def test_abandoned_thread_writes_nothing_when_the_driver_finally_returns(self, monkeypatch):
        entered, release = threading.Event(), threading.Event()

        class HungRecorder:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def record(self, numframes):
                entered.set()
                release.wait(5)  # the frozen driver call
                return np.full((numframes, 1), 0.1, dtype=np.float32)

        monkeypatch.setattr(capture_module.audio, "get_recorder", lambda source: HungRecorder())
        written = []
        thread = capture_module.CaptureThread("mic", on_block=written.append)
        thread.start()
        assert entered.wait(3)
        assert thread.abandon() is None
        release.set()
        thread._thread.join(3)
        assert not thread._thread.is_alive() and written == [] and thread.audio_queue.empty()

    def test_seconds_since_audio_tracks_delivered_blocks(self):
        thread = capture_module.CaptureThread("mic")
        thread.started_at = 100.0
        assert thread.seconds_since_audio(now=112.0) == 12.0
        thread.last_audio_at = 110.0
        assert thread.seconds_since_audio(now=112.0) == 2.0
