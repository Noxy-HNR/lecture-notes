"""Direct tests of stopping a recording: the real run() loop with fake audio, model and notes.

Everything run() touches on disk is redirected into tmp_path - state/ (WAV, segments, raw
log, failure ledger), status.json, command.json, performance timings and the notes file -
and old-backup pruning, sleep blocking and the keypress listener are stubbed out. So these
are safe to run while a real recorder is live: they never read its dashboard commands,
never write its status file and never delete a recording.
"""
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import soundfile as sf

import audio
import main
import performance
import telemetry
import transcribe

SR = audio.SAMPLE_RATE


class FakeModel:
    name = "cohere-transcribe"
    device_desc = "fake/bf16"
    repaired_pieces = 0
    silent_pieces = 0

    def __init__(self):
        self.calls = []

    def transcribe(self, samples, sample_rate):
        seconds = len(samples) / sample_rate
        self.calls.append(round(seconds, 2))
        return [{"text": f"window {len(self.calls)}", "start": 0.0, "end": seconds}]


class FakePreflight:
    def start_audio_check(self, source):
        pass

    def start_gpu_warmup(self, formatting_mode):
        pass

    def finish(self, source, formatting_mode):
        return True


def _eventually(condition, seconds=3.0):
    """Polls with short sleeps, which is also what lets a Python signal handler run."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class FakeCapture:
    """Two 0.5s blocks of speech, then Ctrl+C lands while the third collection holds only
    0.3s. Another 0.2s reaches the queue before capture stops. Each block is persisted
    through on_block first, the way the real capture thread backs audio up before queueing."""

    def __init__(self, source, on_block=None):
        self.on_block = on_block
        self.fatal_error = None
        self.collections = 0
        self.persisted_frames = 0
        self.drained = False

    def _speech(self, seconds):
        block = np.full(int(seconds * SR), 0.1, dtype=np.float32)
        self.on_block(block)
        self.persisted_frames += len(block)
        return block

    def start(self):
        pass

    def stop(self):
        pass

    last_audio_at = 1.0

    def seconds_since_audio(self, now=None):
        return 0.0

    def abandon(self):
        return self.drain_available()

    def poll_errors(self):
        return []

    def collect_chunk(self, target_seconds, interrupt_event=None, **kwargs):
        self.collections += 1
        if self.collections <= 2:
            return self._speech(0.5)
        partial = self._speech(0.3)
        signal.raise_signal(signal.SIGINT)
        assert _eventually(interrupt_event.is_set), "Ctrl+C must wake a part-filled collection"
        return partial

    def drain_available(self):
        if self.drained:
            return None
        self.drained = True
        return self._speech(0.2)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(main, "STATE_DIR", state)
    monkeypatch.setattr(telemetry, "STATE_DIR", state)
    monkeypatch.setattr(telemetry, "STATUS_PATH", state / "status.json")
    monkeypatch.setattr(telemetry, "COMMAND_PATH", state / "command.json")
    monkeypatch.setattr(performance, "PATH", tmp_path / "timings.jsonl")
    monkeypatch.setattr(main, "auto_prune_audio", lambda: None)
    monkeypatch.setattr(main.keep_awake, "prevent_sleep", lambda: None)
    monkeypatch.setattr(main.keep_awake, "allow_sleep", lambda: None)
    monkeypatch.setattr(main, "msvcrt", None)
    monkeypatch.setattr(main, "choose_action", lambda args: "record")
    monkeypatch.setattr(main, "choose_class",
                        lambda args: {"code": "TEST 1000", "title": "Test", "status": "manual"})
    monkeypatch.setattr(main, "PreflightRunner", FakePreflight)
    monkeypatch.setattr(main.notes, "notes_path", lambda code: tmp_path / "notes.md")
    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "mic", "--formatting", "heuristic",
                                      "--chunk", "1"])

    model = FakeModel()
    transcribe.unload_model()
    monkeypatch.setattr(transcribe, "_load_backend", lambda: model)
    captures = []

    def make_capture(source, on_block=None):
        captures.append(FakeCapture(source, on_block))
        return captures[-1]

    monkeypatch.setattr(main.capture_module, "CaptureThread", make_capture)
    yield SimpleNamespace(model=model, captures=captures, saves=[], state=state)
    transcribe.unload_model()


def test_ctrl_c_stops_gracefully_and_keeps_every_captured_frame(recorder, monkeypatch):
    original_handler = signal.getsignal(signal.SIGINT)

    def fake_save(code, title, text, date, mode=None, session_id=None, **kwargs):
        recorder.saves.append((text, session_id))
        return recorder.state / "notes.md", "none"

    monkeypatch.setattr(main.notes, "format_and_save", fake_save)
    main.run()  # returns normally: Ctrl+C must not escape as KeyboardInterrupt mid-bookkeeping

    capture = recorder.captures[0]
    wav = next(recorder.state.glob("TEST_1000_*.wav"))
    assert signal.getsignal(signal.SIGINT) is original_handler
    # One full 1s window, then the 0.3s caught by Ctrl+C plus the 0.2s drained at stop,
    # sent together as a single final window - nothing dropped, nothing transcribed twice.
    assert recorder.model.calls == [1.0, 0.5]
    assert sf.info(str(wav)).frames == capture.persisted_frames == int(1.5 * SR)
    assert recorder.saves == [("window 1 window 2", wav.stem)]
    assert len(wav.with_name(wav.stem + "_segments.jsonl").read_text().splitlines()) == 2
    assert not list(recorder.state.glob("*.failed.json"))


def test_repeated_ctrl_c_during_shutdown_explains_and_does_not_abort_the_save(recorder, monkeypatch, capsys):
    def impatient_save(code, title, text, date, mode=None, session_id=None, **kwargs):
        signal.raise_signal(signal.SIGINT)  # a second press while the final notes are being written
        time.sleep(0.05)
        recorder.saves.append(text)
        return recorder.state / "notes.md", "none"

    monkeypatch.setattr(main.notes, "format_and_save", impatient_save)
    main.run()

    out = capsys.readouterr().out
    assert recorder.saves == ["window 1 window 2"]
    assert out.count("Already stopping") == 1
    assert "All done" in out


def test_stop_request_handler_never_prints_itself(capsys):
    """Shutdown prints constantly. A handler that printed could interrupt an in-progress
    print on the main thread and raise a reentrant-call error, crashing the save."""
    wake = threading.Event()
    stop = main.StopRequest(wake)
    try:
        stop.handle(signal.SIGINT, None)
        assert stop.requested.is_set() and wake.is_set()
        original_print = main.print if hasattr(main, "print") else print
        calls = []
        main.print = lambda *a, **k: calls.append(threading.current_thread().name)
        try:
            stop.handle(signal.SIGINT, None)
            assert _eventually(lambda: calls)
        finally:
            del main.print
        assert threading.main_thread().name not in calls
    finally:
        stop.close()


def _speech(seconds):
    return np.full(int(seconds * SR), 0.1, dtype=np.float32)


class DropoutCapture(FakeCapture):
    """4s of exact zeros (a muted or disconnected mic), then speech, then Ctrl+C."""

    def collect_chunk(self, target_seconds, interrupt_event=None, **kwargs):
        self.collections += 1
        if self.collections <= 8:
            block = np.zeros(int(0.5 * SR), dtype=np.float32)
            self.on_block(block)
            self.persisted_frames += len(block)
            return block
        if self.collections == 9:
            return self._speech(0.5)
        partial = self._speech(0.3)
        signal.raise_signal(signal.SIGINT)
        assert _eventually(interrupt_event.is_set)
        return partial


def test_mic_dropout_is_announced_within_seconds_and_when_it_returns(recorder, monkeypatch, capsys):
    monkeypatch.setattr(main.notes, "format_and_save", lambda *a, **k: (recorder.state / "notes.md", "none"))
    monkeypatch.setattr(main.mic_mute, "is_muted", lambda: False)
    monkeypatch.setattr(main.capture_module, "CaptureThread",
                        lambda source, on_block=None: DropoutCapture(source, on_block))
    main.run()

    out = capsys.readouterr().out
    assert out.count("Microphone signal lost") == 1
    assert "Microphone signal is back after 4s" in out
    messages = [e["message"] for e in telemetry.read_status()["events"]]
    assert any("signal lost" in m for m in messages) and any("is back" in m for m in messages)


class FreezingCapture(FakeCapture):
    """The first capture delivers 1s of speech, then its device freezes after sleep: no
    audio and no error, so collection just times out. The replacement works normally."""
    made = []

    def __init__(self, source, on_block=None):
        super().__init__(source, on_block)
        self.number = len(FreezingCapture.made) + 1
        self.frozen = False
        FreezingCapture.made.append(self)

    def seconds_since_audio(self, now=None):
        return 999.0 if self.frozen else 0.0

    def collect_chunk(self, target_seconds, interrupt_event=None, **kwargs):
        if self.number != 1:
            return super().collect_chunk(target_seconds, interrupt_event, **kwargs)
        self.collections += 1
        if self.collections <= 2:
            return self._speech(0.5)
        self.frozen = True
        return None


def test_frozen_audio_device_is_reconnected_without_losing_audio(recorder, monkeypatch, capsys):
    FreezingCapture.made = []
    saves = []
    monkeypatch.setattr(main.notes, "format_and_save",
                        lambda code, title, text, *a, **k: saves.append(text) or (recorder.state / "notes.md", "none"))
    monkeypatch.setattr(main.capture_module, "CaptureThread",
                        lambda source, on_block=None: FreezingCapture(source, on_block))
    main.run()

    out = capsys.readouterr().out
    first, second = FreezingCapture.made
    assert len(FreezingCapture.made) == 2 and out.count("Reconnecting") == 1
    assert "coming in again" in out
    wav = next(recorder.state.glob("TEST_1000_*.wav"))
    assert sf.info(str(wav)).frames == first.persisted_frames + second.persisted_frames
    # Nothing captured before the freeze or after the reconnect is missing from the notes.
    assert sum(recorder.model.calls) == pytest.approx((first.persisted_frames + second.persisted_frames) / SR)
    # The frozen capture still had 0.2s queued; it's recovered into the next window (0.2 + 1.0).
    assert recorder.model.calls == [1.0, 1.2, 0.5] and saves == ["window 1 window 2 window 3"]
    messages = [e["message"] for e in telemetry.read_status()["events"]]
    assert any("stopped responding" in m for m in messages)


def test_transcript_guard_activity_reaches_the_dashboard(recorder, monkeypatch):
    model, original = recorder.model, recorder.model.transcribe

    def guarded(samples, sample_rate):
        result = original(samples, sample_rate)
        if len(model.calls) == 1:
            model.silent_pieces += 1
        else:
            model.repaired_pieces += 1
        return result

    model.transcribe = guarded
    monkeypatch.setattr(main.notes, "format_and_save", lambda *a, **k: (recorder.state / "notes.md", "none"))
    main.run()

    status = telemetry.read_status()
    assert (status["guard_repaired"], status["guard_silent"]) == (1, 1)
    messages = [e["message"] for e in status["events"]]
    assert sum("looping piece" in m for m in messages) == 1
    assert sum("silent piece" in m for m in messages) == 1


class TestDropoutMonitor:
    def test_quiet_room_is_not_a_dropout(self):
        monitor = main.DropoutMonitor(SR)
        room = np.random.default_rng(0).normal(0, 0.0005, 60 * SR).astype(np.float32)
        assert monitor.feed(room) == [] and monitor.dropout_seconds == 0

    def test_warns_once_after_three_seconds_across_slices_then_reports_return(self):
        monitor = main.DropoutMonitor(SR, muted_check=lambda: True)
        zeros = np.zeros(int(2.5 * SR), dtype=np.float32)
        assert monitor.feed(zeros) == []  # 2.5s: not yet
        lost = monitor.feed(zeros)        # 5s, spanning two collected slices
        assert [level for level, _ in lost] == ["error"]
        assert "Windows has the microphone muted" in lost[0][1]
        assert monitor.feed(zeros) == []  # still out: no repeated warning
        assert monitor.feed(_speech(1.0)) == [
            ("success", "Microphone signal is back after 8s - that stretch wasn't recorded.")]
        assert monitor.dropout_seconds == 0

    def test_brief_glitch_is_ignored(self):
        monitor = main.DropoutMonitor(SR)
        assert monitor.feed(np.concatenate([np.zeros(2 * SR, dtype=np.float32), _speech(1.0)])) == []

    def test_unreadable_mute_state_gives_the_generic_message(self):
        def broken():
            raise OSError("COM unavailable")
        lost = main.DropoutMonitor(SR, muted_check=broken).feed(np.zeros(4 * SR, dtype=np.float32))
        assert "muted or disconnected" in lost[0][1]


def test_autosave_and_manual_save_keep_snapshot_order(monkeypatch):
    """Autosave runs on the main loop and a manual save on the transcription worker. If the
    earlier snapshot's thread is preempted between taking its text and queueing the save,
    the later snapshot must not jump ahead - it would write newer notes above older ones."""

    class PreemptedSession:
        session_id = "TEST_1000_20260914_000000"

        def __init__(self):
            self.snapshots = 0
            self.first_taken = threading.Event()
            self.resume_first = threading.Event()

        def pop_pending_text(self, run_diarization=False):
            self.snapshots += 1
            if self.snapshots == 1:
                self.first_taken.set()
                assert self.resume_first.wait(3)
                return "earlier speech"
            return "later speech"

    saved = []
    monkeypatch.setattr(main.notes, "format_and_save",
                        lambda *args, **kwargs: (saved.append(args[2]), (Path("notes.md"), "none"))[1])
    cls = {"code": "TEST 1000", "title": "Test"}
    session = PreemptedSession()
    worker = main.SaveWorker()

    autosave = threading.Thread(target=main._save, args=(cls, session, "date", "heuristic"),
                                kwargs={"note": "(autosave)", "save_worker": worker})
    autosave.start()
    assert session.first_taken.wait(3)
    manual = threading.Thread(target=main._save, args=(cls, session, "date", "heuristic"),
                              kwargs={"note": "(manual save)", "save_worker": worker})
    manual.start()
    manual.join(0.2)  # without ordering, the later snapshot gets queued during this pause
    session.resume_first.set()
    autosave.join(3)
    manual.join(3)
    worker.wait_idle()
    assert saved == ["earlier speech", "later speech"]
