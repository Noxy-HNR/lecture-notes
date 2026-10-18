"""Discontinuity ownership must not change warning behavior in other threads."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
import warnings

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audio
import capture


def test_warning_counter_is_thread_local_and_other_warnings_survive():
    class Gap(RuntimeWarning):
        pass
    router = audio._SoundcardWarnings(warnings, Gap)
    counts = []
    with audio.count_discontinuities(lambda: counts.append(1)):
        router.warn("data discontinuity in recording", Gap)
        with pytest.warns(UserWarning, match="another warning"):
            router.warn("another warning", UserWarning)
        with pytest.warns(Gap):
            thread = threading.Thread(target=lambda: router.warn("data discontinuity in recording", Gap))
            thread.start()
            thread.join()
    assert counts == [1]
    with pytest.warns(Gap):
        router.warn("data discontinuity in recording", Gap)


def test_counter_restores_after_failure():
    original, inner = [], []
    router = audio._SoundcardWarnings(warnings, RuntimeWarning)
    with audio.count_discontinuities(lambda: original.append(1)):
        with pytest.raises(ValueError), audio.count_discontinuities(lambda: inner.append(1)):
            router.warn("data discontinuity in recording", RuntimeWarning)
            raise ValueError()
        router.warn("data discontinuity in recording", RuntimeWarning)
    assert original == inner == [1]


@pytest.mark.skipif(not hasattr(audio.sc, "mediafoundation"), reason="WASAPI warning hook")
def test_installed_wasapi_hook_counts_without_changing_global_filters():
    before = list(warnings.filters)
    counted = []
    native = audio.sc.mediafoundation
    with audio.count_discontinuities(lambda: counted.append(1)):
        native.warnings.warn("data discontinuity in recording", native.SoundcardRuntimeWarning)
    assert counted == [1] and warnings.filters == before


def test_capture_counts_gap_without_losing_persisted_block(monkeypatch):
    import numpy as np
    thread = capture.CaptureThread("mic")
    stored = []
    thread.on_block = stored.append
    router = audio._SoundcardWarnings(warnings, RuntimeWarning)
    class Recorder:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def record(self, numframes):
            router.warn("data discontinuity in recording", RuntimeWarning)
            thread._stop_event.set()
            return np.ones((numframes, 1), dtype=np.float32)
    monkeypatch.setattr(audio, "get_recorder", lambda source: Recorder())
    thread._run()
    assert thread.discontinuities == 1
    assert len(stored) == 1 and thread.audio_queue.qsize() == 1


def test_stereo_request_preserves_default_mono(monkeypatch):
    requests = []
    device = SimpleNamespace(recorder=lambda **kwargs: requests.append(kwargs))
    monkeypatch.setattr(audio, "pick_microphone", lambda: device)
    audio.get_recorder("mic")
    audio.get_recorder("mic", channels=2)
    assert [r["channels"] for r in requests] == [1, 2]


def test_watchdog_preserves_gap_count_across_reconnect():
    import main
    clock = [0.0]
    captures = []
    class FakeCapture:
        last_audio_at = None
        discontinuities = 3
        def start(self): pass
        def seconds_since_audio(self, now): return now
        def abandon(self): return None
    def make():
        instance = FakeCapture()
        captures.append(instance)
        return instance
    watchdog = main.CaptureWatchdog(make, clock=lambda: clock[0])
    clock[0] = 40
    watchdog.check()
    assert watchdog.restarts == 1
    captures[1].discontinuities = 2
    assert watchdog.discontinuities == 5
