"""Isolated checks: no recorder process, GPU inference, or live state files."""
import threading
import numpy as np
import main
import telemetry


def test_save_barrier_includes_requested_audio_before_later_audio():
    entered, release = threading.Event(), threading.Event()
    seen = []
    class FakeTranscriber:
        def process(self, samples):
            entered.set()
            assert release.wait(3)
            return [len(samples)]
    worker = main.TranscriptionWorker(FakeTranscriber(), lambda a, s, t: seen.append(s[0]))
    worker.submit(np.zeros(10))
    assert entered.wait(3)
    worker.after_pending(lambda: seen.append(tuple(seen)))
    worker.submit(np.zeros(20))
    release.set()
    worker.wait_idle()
    assert seen == [10, (10,), 20]


def test_heartbeat_runs_without_transcription_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(telemetry, 'STATUS_PATH', tmp_path / 'status.json')
    tel = telemetry.Telemetry()
    # Accelerate only this instance's heartbeat wait; no real recording involved.
    original_wait = tel._heartbeat_stop.wait
    monkeypatch.setattr(tel._heartbeat_stop, 'wait', lambda seconds: original_wait(0.01))
    written = threading.Event()
    original_flush = tel.flush
    def flush():
        original_flush()
        if threading.current_thread().name == 'notes-heartbeat':
            written.set()
    monkeypatch.setattr(tel, 'flush', flush)
    tel.start_session(class_code='TEST')
    try:
        assert written.wait(3)
        assert telemetry.read_status()['active'] is True
    finally:
        tel.end_session()
    assert not tel._heartbeat_thread.is_alive()
    assert telemetry.read_status()['active'] is False
