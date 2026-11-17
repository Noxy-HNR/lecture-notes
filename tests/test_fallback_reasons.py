"""When Claude and the local model can't format a lecture, the notes say why; and a cleanup
stopped by Claude's usage limit waits for the reset time and tries again."""
import json
import time
from datetime import datetime

import pytest

import gpu_formatter
import notes
import repair

LIMIT = "the Claude CLI exited with code 1: You've hit your 5-hour limit · resets 11:50am (America/Chicago)"


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "notes_path", lambda code: tmp_path / f"{code.replace(' ', '_')}.md")
    monkeypatch.setattr(notes, "_proofread", lambda title, code, text: text)
    return tmp_path


def claude_down(monkeypatch, failure=LIMIT):
    def down(prompt, timeout=None, **kwargs):
        notes.last_cli_failure = failure
        return None
    monkeypatch.setattr(notes, "_try_claude_cli_format", down)
    monkeypatch.setattr(notes, "_try_claude_api_format", lambda *a, **k: None)


def local_model_refuses(monkeypatch, why):
    def refuse(*args):
        gpu_formatter._last_error = why
        return None
    monkeypatch.setattr(gpu_formatter, "format_transcript", refuse)


# --- the reason, in the notes and on the console ------------------------------------------------

def test_basic_notes_say_why_claude_and_the_local_model_were_not_used(notes_dir, monkeypatch):
    claude_down(monkeypatch)
    local_model_refuses(monkeypatch, gpu_formatter._ram_reason(1.8 * 2**30, 3.3 * 2**30))
    path, method = notes.format_and_save("BIOL 1440", "Biology", "cells divide by mitosis. " * 20,
                                         "Tuesday, November 17, 2026", mode="auto", session_id="BIOL_1440_x")
    assert method == "local"
    text = path.read_text(encoding="utf-8")
    line = ("*(auto-formatted locally — Claude's usage limit was reached (resets 11:50am); the local model "
            "was skipped: only 1.8 GB of RAM was free and the local model needs 3.3 GB)*")
    assert line in text
    assert notes.HEURISTIC_MARKER in text, "the cleanup job still recognises it as rough"
    assert notes.last_fallback_reason.startswith("Claude's usage limit was reached")


def test_local_model_notes_say_why_claude_was_not_used(notes_dir, monkeypatch):
    claude_down(monkeypatch, "the Claude CLI exited with code 1: not logged in")
    monkeypatch.setattr(gpu_formatter, "format_transcript", lambda *a: "## Tuesday\n\n- model notes")
    _, method = notes.format_and_save("BIOL 1440", "Biology", "words " * 40, "Tuesday", mode="auto")
    assert method == "gpu"
    assert notes.last_fallback_reason == "Claude couldn't be reached: exited with code 1: not logged in"


def test_no_reason_when_claude_wrote_the_notes_or_heuristic_was_chosen(notes_dir, monkeypatch):
    monkeypatch.setattr(notes, "_try_claude_cli_format", lambda prompt, timeout=None, **k: "## Tuesday\n\n- fine")
    notes.format_and_save("BIOL 1440", "Biology", "words " * 40, "Tuesday", mode="auto")
    assert notes.last_fallback_reason is None
    path, method = notes.format_and_save("CHEM 1450", "Chemistry", "words " * 40, "Tuesday", mode="heuristic")
    assert method == "local" and notes.last_fallback_reason is None
    assert "*(auto-formatted locally — Claude CLI/API unavailable)*" in path.read_text(encoding="utf-8")


def test_local_only_mode_says_why_the_local_model_was_skipped(notes_dir, monkeypatch):
    monkeypatch.setattr(notes, "_try_claude_cli_format", lambda *a, **k: pytest.fail("local mode called Claude"))
    local_model_refuses(monkeypatch, "another local model process is still running and has to exit first")
    path, _ = notes.format_and_save("BIOL 1440", "Biology", "words " * 40, "Tuesday", mode="local")
    assert ("— the local model was skipped: another local model process is still running and has to exit first)*"
            in path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("failure, summary", [
    (LIMIT, "Claude's usage limit was reached (resets 11:50am)"),
    ("the Claude CLI exited with code 1: You've hit your weekly limit", "Claude's usage limit was reached"),
    ("the Claude CLI did not finish within 15 minutes", "Claude couldn't be reached: did not finish within 15 minutes"),
    (None, "Claude couldn't be reached"),
])
def test_claude_failures_are_summarised(failure, summary):
    assert notes.claude_failure_summary(failure) == summary


def test_the_recorder_prints_and_logs_the_reason(notes_dir, monkeypatch, capsys):
    import main

    class Tel:
        events = []

        def record_save(self, *a):
            pass

        def add_event(self, level, message):
            self.events.append(message)

        def flush(self):
            pass

    class FakeSession:
        session_id = "BIOL_1440_x"

        def take_pending(self, run_diarization=False):
            return "cells divide. " * 30, None

        def note_saved(self, method):
            pass

    claude_down(monkeypatch)
    local_model_refuses(monkeypatch, "only 1.8 GB of RAM was free and the local model needs 3.3 GB")
    tel = Tel()
    assert main._save({"code": "BIOL 1440", "title": "Biology"}, FakeSession(), "Tuesday", "auto", telemetry=tel)
    out = capsys.readouterr().out
    assert ("[heuristic local formatting - Claude's usage limit was reached (resets 11:50am); the local model "
            "was skipped: only 1.8 GB of RAM was free and the local model needs 3.3 GB]") in out
    assert "only 1.8 GB of RAM was free" in tel.events[-1]


# --- the local model's own reasons -----------------------------------------------------------------

def test_the_local_model_reports_low_ram_with_the_numbers(monkeypatch):
    monkeypatch.setattr(gpu_formatter, "_saved_profile", lambda: ({}, int(3.3 * 2**30)))
    monkeypatch.setattr(gpu_formatter, "_port_open", lambda: False)
    monkeypatch.setattr(gpu_formatter, "_llama_process_ids", lambda: [])
    monkeypatch.setattr(gpu_formatter, "_available_physical_bytes", lambda: int(1.84 * 2**30))
    monkeypatch.setattr(gpu_formatter, "_ensure_manager", lambda minimum: pytest.fail("must not start anything"))
    assert gpu_formatter._ensure_server_running() is False
    assert gpu_formatter.last_failure() == "only 1.8 GB of RAM was free and the local model needs 3.3 GB"


def test_a_successful_local_format_clears_the_reason(monkeypatch):
    gpu_formatter._last_error = "old reason"
    monkeypatch.setattr(gpu_formatter, "_chat", lambda *a, **k: "- point one")
    assert gpu_formatter.format_transcript("Biology", "Tuesday", "words") == "## Tuesday\n\n- point one"
    assert gpu_formatter.last_failure() is None


def test_a_malformed_local_reply_says_so(monkeypatch):
    monkeypatch.setattr(gpu_formatter, "_chat", lambda *a, **k: "# Invented title\n\n- point")
    assert gpu_formatter.format_transcript("Biology", "Tuesday", "words") is None
    assert "unexpected layout" in gpu_formatter.last_failure()


# --- the cleanup waits for the usage limit to reset ---------------------------------------------

def at(hour, minute=0):
    return datetime(2026, 11, 17, hour, minute).timestamp()


@pytest.mark.parametrize("message, now, expected", [
    (LIMIT, at(9, 30), at(11, 52)),
    ("You've hit your limit · resets 3pm (America/Chicago)", at(9), at(15, 2)),
    ("You've hit your limit · resets 12am", at(23), at(0, 2) + 86400),
    ("You've hit your limit · resets 11:50am", at(11, 55), at(11, 52)),       # just passed: retry now
    ("You've hit your limit · resets 9am", at(22), None),                     # 11 h away: misread
    ("the Claude CLI exited with code 1: not logged in", at(9), None),
    ("hit your limit, try later", at(9), None),
    (None, at(9), None),
])
def test_limit_reset_time_is_read_from_the_message(message, now, expected):
    assert repair.limit_reset_at(message, now) == expected


@pytest.fixture
def cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "STATE_DIR", tmp_path)
    monkeypatch.setattr(repair, "JOB_PATH", tmp_path / "repair_job.json")
    monkeypatch.setattr(repair, "MEMORY_PATH", tmp_path / "repair_memory.json")
    monkeypatch.setattr(repair, "recording_active", lambda: False)
    monkeypatch.setattr(repair, "_housekeeping", lambda: None)
    monkeypatch.setattr(repair, "find_unit", lambda content, key: {"text": "notes", "kind": "session"})
    monkeypatch.setattr(repair, "notes_file", lambda code: tmp_path / "notes.md")
    (tmp_path / "notes.md").write_text("notes", encoding="utf-8")
    monkeypatch.setattr(repair, "WAIT_POLL_SECONDS", 0)
    clock = {"now": at(9, 0)}
    monkeypatch.setattr(repair.time, "time", lambda: clock["now"])
    monkeypatch.setattr(repair.time, "sleep", lambda s: clock.update(now=clock["now"] + 60))
    return clock


def job_with(*keys, trigger="after_lecture"):
    items = [{"class_code": "BIOL 1440", "key": k, "label": k, "action": "fix", "state": "queued", "message": ""}
             for k in keys]
    job = {"id": "j1", "status": "running", "trigger": trigger, "created_at": time.time(), "items": items}
    repair._write_json(repair.JOB_PATH, job)
    return job


def read_job():
    return json.loads(repair.JOB_PATH.read_text(encoding="utf-8"))


def test_a_limited_cleanup_waits_for_the_reset_and_finishes_the_rest(cleanup, monkeypatch):
    job_with("a", "b", "c")
    calls, seen_waiting = [], []

    def run(item):
        calls.append((item["key"], cleanup["now"]))
        if cleanup["now"] < at(10, 0):
            return {"ok": False, "claude_down": True, "message": "Claude couldn't be reached (" + LIMIT + ")"}
        return {"ok": True, "message": "Re-formatted"}

    def watching_sleep(seconds):
        state = read_job()
        if state["status"] == "waiting":
            seen_waiting.append(state)
        cleanup["now"] += 60
    monkeypatch.setattr(repair, "_run_item", run)
    monkeypatch.setattr(repair.time, "sleep", watching_sleep)
    monkeypatch.setattr(repair, "limit_reset_at", lambda message, now=None: at(10, 0) if "limit" in message else None)

    assert repair.run_job("j1") == 0
    assert calls[0] == ("a", at(9, 0)), "the first lecture hit the limit"
    assert [key for key, when in calls[1:]] == ["a", "b", "c"], "all three were done after the reset"
    assert all(when >= at(10, 0) for _, when in calls[1:])
    waiting = seen_waiting[0]
    assert waiting["message"] == "Claude's usage limit was reached; the rest is tried again at 10:00 AM"
    assert [i["state"] for i in waiting["items"]] == ["queued", "queued", "queued"]
    final = read_job()
    assert final["status"] == "finished" and final["message"] == "3 of 3 lecture(s) fixed"
    assert [i["state"] for i in final["items"]] == ["done", "done", "done"]


def test_a_failure_that_is_not_the_limit_does_not_wait(cleanup, monkeypatch):
    job_with("a", "b")
    monkeypatch.setattr(repair, "_run_item", lambda item: {"ok": False, "claude_down": True,
                                                           "message": "Claude couldn't be reached (not logged in)"})
    monkeypatch.setattr(repair.time, "sleep", lambda s: pytest.fail("waited for no reason"))
    repair.run_job("j1")
    final = read_job()
    assert final["status"] == "finished" and "isn't reachable" in final["message"]
    assert [i["state"] for i in final["items"]] == ["failed", "skipped"]


def test_waiting_gives_up_after_a_few_resets(cleanup, monkeypatch):
    job_with("a")
    monkeypatch.setattr(repair, "_run_item", lambda item: {"ok": False, "claude_down": True,
                                                           "message": "Claude couldn't be reached (" + LIMIT + ")"})
    monkeypatch.setattr(repair, "limit_reset_at", lambda message, now=None: cleanup["now"] + 120)
    repair.run_job("j1")
    final = read_job()
    assert final["status"] == "finished" and final["items"][0]["state"] == "failed"


def test_the_retry_waits_for_a_recording_to_end(cleanup, monkeypatch):
    job_with("a")
    results = iter([{"ok": False, "claude_down": True, "message": LIMIT}, {"ok": True, "message": "done"}])
    ran_at = []
    monkeypatch.setattr(repair, "_run_item", lambda item: ran_at.append(cleanup["now"]) or next(results))
    monkeypatch.setattr(repair, "limit_reset_at", lambda message, now=None: at(10, 0))
    monkeypatch.setattr(repair, "recording_active", lambda: at(9, 30) <= cleanup["now"] < at(11, 0))
    repair.run_job("j1")
    assert ran_at[1] >= at(11, 0)
    assert read_job()["status"] == "finished"


def test_cancelling_a_waiting_cleanup_stops_it(cleanup, monkeypatch):
    job_with("a", "b")
    monkeypatch.setattr(repair, "_run_item", lambda item: {"ok": False, "claude_down": True, "message": LIMIT})
    monkeypatch.setattr(repair, "limit_reset_at", lambda message, now=None: at(10, 0))
    monkeypatch.setattr(repair, "_job_alive", lambda job: True)

    def cancel_while_waiting(seconds):
        if read_job()["status"] == "waiting":
            assert repair.cancel_job()["ok"]
        cleanup["now"] += 60
    monkeypatch.setattr(repair.time, "sleep", cancel_while_waiting)
    repair.run_job("j1")
    final = read_job()
    assert final["status"] == "cancelled" and "won't try again" in final["message"]
    assert [i["state"] for i in final["items"]] == ["skipped", "skipped"]


def test_a_new_cleanup_replaces_a_waiting_one_and_keeps_its_work(cleanup, monkeypatch):
    monkeypatch.setattr(repair, "_job_alive", lambda job: True)
    spawned = []
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    waiting = job_with("old-lecture", "done-lecture")
    waiting["items"][1]["state"] = "done"
    waiting.update(status="waiting", retry_at=at(10, 0), pid=123)
    repair._write_json(repair.JOB_PATH, waiting)
    new = [{"class_code": "BIOL 1440", "key": "new-lecture", "label": "new", "action": "fix",
            "state": "queued", "message": ""}]
    job = repair._spawn(new, "after_lecture")
    assert [i["key"] for i in job["items"]] == ["new-lecture", "old-lecture"]
    assert job["id"] != "j1" and spawned
    # The waiting process notices it was replaced and leaves the new job alone.
    assert repair._wait_for_reset({**waiting, "id": "j1"}, at(10, 0)) == "gone"
    assert read_job()["id"] == job["id"] and read_job()["status"] == "running"
