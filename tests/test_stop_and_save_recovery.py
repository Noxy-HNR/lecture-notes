"""A leftover Stop must never end the next recording, a failed save must never leave a hole in
the notes, and --captions auto must follow the power cable during a lecture."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

import storage
import telemetry


@pytest.fixture
def tel_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "STATE_DIR", tmp_path)
    monkeypatch.setattr(telemetry, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(telemetry, "COMMAND_PATH", tmp_path / "command.json")
    return tmp_path


# -- command channel -------------------------------------------------------------------------

def test_a_stop_left_from_an_earlier_session_is_ignored(tel_dir):
    telemetry.COMMAND_PATH.write_text(json.dumps({"action": "stop", "at": time.time() - 3600}), encoding="utf-8")
    assert telemetry.take_command() is None
    assert not telemetry.COMMAND_PATH.exists()


def test_a_fresh_command_is_taken_once(tel_dir):
    assert telemetry.send_command("stop")
    assert telemetry.take_command() == "stop"
    assert telemetry.take_command() is None


def test_a_command_without_a_time_is_ignored(tel_dir):
    telemetry.COMMAND_PATH.write_text(json.dumps({"action": "stop"}), encoding="utf-8")
    assert telemetry.take_command() is None


def test_starting_a_session_clears_a_waiting_command(tel_dir):
    assert telemetry.send_command("stop")
    tel = telemetry.Telemetry()
    tel.start_session(class_code="TEST 100", class_title="Test")
    try:
        assert telemetry.take_command() is None
    finally:
        tel.end_session()


def test_stopping_is_reported_and_ending_clears_commands(tel_dir):
    tel = telemetry.Telemetry()
    tel.start_session(class_code="TEST 100", class_title="Test")
    try:
        assert telemetry.read_status()["stopping"] is False
        tel.mark_stopping()
        assert telemetry.read_status()["stopping"] is True
        telemetry.send_command("stop")  # e.g. a second click while the notes save
    finally:
        tel.end_session()
    status = telemetry.read_status()
    assert status["active"] is False and status["stopping"] is False
    assert not telemetry.COMMAND_PATH.exists()


def test_commands_are_refused_unless_a_live_recording_can_take_them():
    now = time.time()
    assert telemetry.command_refusal({}) is not None
    assert telemetry.command_refusal({"active": False, "updated_at": now}) is not None
    assert telemetry.command_refusal({"active": True, "updated_at": now - 120}) is not None
    assert "already stopping" in telemetry.command_refusal({"active": True, "updated_at": now, "stopping": True})
    assert telemetry.command_refusal({"active": True, "updated_at": now, "stopping": False}) is None


def test_diagnostics_page_keeps_stop_off_while_stopping():
    page = (Path(__file__).resolve().parent.parent / "dashboard/diagnostics.html").read_text(encoding="utf-8")
    assert "!!data.stopping" in page
    assert "document.getElementById('btn-stop').disabled = !active || stale || stopping;" in page


# -- save recovery ---------------------------------------------------------------------------

def _session_with_text(tmp_path, *texts):
    import main
    session = main.Session(tmp_path / "s.wav")
    for text in texts:
        session.write_chunk(np.zeros(160, dtype=np.float32), [{"start": 0.0, "end": 0.1, "text": text}])
    return session


def test_restored_text_comes_back_before_newer_text(tmp_path):
    session = _session_with_text(tmp_path, "first")
    try:
        text, taken = session.take_pending()
        assert text == "first" and not session.has_pending()
        session.write_chunk(np.zeros(160, dtype=np.float32), [{"start": 0.0, "end": 0.1, "text": "second"}])
        session.restore_pending(taken)
        assert session.pop_pending_text() == "first second"
    finally:
        session.close()


def test_a_failed_save_keeps_its_text_for_the_next_save(tmp_path, monkeypatch):
    import main
    session = _session_with_text(tmp_path, "the lecture so far")
    calls = []

    def fail_then_save(code, title, text, date, **kwargs):
        calls.append(text)
        if len(calls) == 1:
            raise PermissionError("notes file is open in another program")
        return Path("notes.md"), "none"

    monkeypatch.setattr(main.notes, "format_and_save", fail_then_save)
    cls = {"code": "TEST 1000", "title": "Test"}
    try:
        assert main._save(cls, session, "date", "heuristic") is False
        assert session.has_pending()
        assert main._save(cls, session, "date", "heuristic") is True
        assert calls == ["the lecture so far", "the lecture so far"]
        assert not session.has_pending()
    finally:
        session.close()


def test_a_failed_background_save_keeps_its_text(tmp_path, monkeypatch):
    import main
    session = _session_with_text(tmp_path, "autosaved words")

    def fail(*args, **kwargs):
        raise TimeoutError("File is busy; try saving again")

    monkeypatch.setattr(main.notes, "format_and_save", fail)
    worker = main.SaveWorker()
    try:
        assert main._save({"code": "TEST 1000", "title": "Test"}, session, "date", "heuristic",
                          save_worker=worker) is None
        worker.wait_idle()
        assert session.pop_pending_text() == "autosaved words"
    finally:
        session.close()


def test_a_failed_save_stays_ahead_of_a_later_window_already_queued(tmp_path, monkeypatch):
    """Autosave A fails after a manual save B has already taken the next window. B must write
    A's text first, so the notes keep lecture order."""
    import threading
    import main
    session = main.Session(tmp_path / "s.wav")
    session.record_segments([{"start": 0.0, "end": 1.0, "text": "first"}], 0.0)
    written, release_a = [], threading.Event()

    def save(code, title, text, date, **kwargs):
        if not written and not release_a.is_set():
            release_a.wait(3)
            written.append(None)  # A's attempt
            raise PermissionError("notes file is open in another program")
        written.append(text)
        return Path("notes.md"), "gpu"

    monkeypatch.setattr(main.notes, "format_and_save", save)
    cls = {"code": "TEST 1000", "title": "Test"}
    worker = main.SaveWorker()
    try:
        main._save(cls, session, "date", "local", save_worker=worker)          # A takes "first"
        session.record_segments([{"start": 0.0, "end": 1.0, "text": "second"}], 60.0)
        main._save(cls, session, "date", "local", save_worker=worker)          # B takes "second"
        release_a.set()
        worker.wait_idle()
        assert written[1:] == ["first second"]
        assert not session.has_pending()
        assert session.best_save_method == "gpu"
    finally:
        session.close()


def test_two_failed_saves_come_back_in_lecture_order(tmp_path):
    import main
    session = main.Session(tmp_path / "s.wav")
    try:
        session.record_segments([{"start": 0.0, "end": 1.0, "text": "first"}], 0.0)
        _, a = session.take_pending()
        session.record_segments([{"start": 0.0, "end": 1.0, "text": "second"}], 60.0)
        _, b = session.take_pending()
        session.restore_pending(a)
        session.restore_pending(b)
        assert session.pop_pending_text() == "first second"
    finally:
        session.close()


def test_replace_retries_while_another_program_holds_the_file(tmp_path, monkeypatch):
    real_replace = storage.os.replace
    attempts = []

    def busy_twice(source, destination):
        attempts.append(destination)
        if len(attempts) <= 2:
            raise PermissionError("sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", busy_twice)
    monkeypatch.setattr(storage, "REPLACE_RETRY_SECONDS", 0)
    target = tmp_path / "notes.md"
    storage.atomic_write(target, "saved", backup=False)
    assert target.read_text(encoding="utf-8") == "saved"
    assert len(attempts) == 3


def test_replace_gives_up_after_its_attempts(tmp_path, monkeypatch):
    def always_busy(source, destination):
        raise PermissionError("sharing violation")

    monkeypatch.setattr(storage.os, "replace", always_busy)
    monkeypatch.setattr(storage, "REPLACE_RETRY_SECONDS", 0)
    with pytest.raises(PermissionError):
        storage.atomic_write(tmp_path / "notes.md", "saved", backup=False)


# -- captions and power ----------------------------------------------------------------------

def test_auto_captions_recheck_power_during_the_lecture():
    source = (Path(__file__).resolve().parent.parent / "src/main.py").read_text(encoding="utf-8")
    loop = source[source.index("while True:\n            chunk_audio = capture.collect_chunk"):]
    loop = loop[:loop.index("except KeyboardInterrupt")]
    assert "args.captions == 'auto'" in loop
    assert "follow_power_for_captions()" in loop
