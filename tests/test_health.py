"""System health: every probe is faked, nothing real is read, spawned or called.

The Claude CLI, RAM, disk, sockets, the lid watcher and the narration voice are all replaced, and the
project folders point at tmp_path. The model must never be reached by snapshot(): the default fake
of notes._try_claude_cli_format records every call, and the tests that poll assert that none was made.
(It can't simply raise: snapshot() catches each probe's errors, so a raise would be hidden.)

Note: call health.test_claude as an attribute - importing it by name would make pytest collect it.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from collections import namedtuple
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import dashboard
import health
import lesson_cleanup
import lessons
import lid_watch
import notes
import speech
import telemetry

GIB = 1024 ** 3
DAY = 86400
SECTIONS = ["claude", "memory", "disk", "recorder", "lesson_build", "narration", "lid_watch", "llama_server", "storage"]


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeLid:
    instances, result = [], True

    def __init__(self, on_close, on_open=None):
        self.stopped = False
        FakeLid.instances.append(self)

    def start(self, timeout=3.0):
        return FakeLid.result

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def calm_world(tmp_path, monkeypatch):
    """A healthy machine, empty project folders and nothing that can reach a real service."""
    state, notes_dir, lessons_dir = tmp_path / "state", tmp_path / "notes", tmp_path / "lessons"
    for folder in (state, notes_dir, lessons_dir / ".jobs"):
        folder.mkdir(parents=True)
    monkeypatch.setattr(health, "STATE_DIR", state)
    monkeypatch.setattr(health, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(health, "LESSONS_DIR", lessons_dir)
    monkeypatch.setattr(lessons, "LESSONS_DIR", lessons_dir)
    monkeypatch.setattr(lessons, "_processes", {})
    monkeypatch.setattr(lessons, "_builder_alive", lambda job: job.get("id") in world.alive)
    monkeypatch.setattr(notes, "_find_claude_cli", lambda: "C:/fake/claude.exe")
    monkeypatch.setattr(notes, "last_cli_failure", None)

    def no_model(prompt, timeout=0):
        world.model_calls.append(prompt)
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(notes, "_try_claude_cli_format", no_model)
    monkeypatch.setattr(health.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 0, "2.1.4 (Claude Code)\n", ""))
    monkeypatch.setattr(health, "_read_memory", lambda: (8 * GIB, 15 * GIB))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(health.shutil, "disk_usage", lambda path: usage(500 * GIB, 300 * GIB, 200 * GIB))
    monkeypatch.setattr(speech, "available", lambda: (True, ""))
    FakeLid.instances, FakeLid.result = [], True
    monkeypatch.setattr(lid_watch, "LidWatcher", FakeLid)

    # health.socket is the socket module itself, so this patch is process-wide: the HTTP client in the
    # dashboard tests opens its connections through the same function. Only the model server's port
    # is refused; everything else gets the real connect.
    real_connect = socket.create_connection

    def refuse_model_server(address, *args, **kwargs):
        if tuple(address) == ("127.0.0.1", health.LLAMA_PORT):
            raise ConnectionRefusedError()
        return real_connect(address, *args, **kwargs)

    monkeypatch.setattr(health.socket, "create_connection", refuse_model_server)
    monkeypatch.setattr(telemetry, "read_status", lambda: {})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    world.alive = set()
    world.model_calls = []
    health.reset_caches()
    yield
    health.reset_caches()


world = types.SimpleNamespace(alive=set(), model_calls=[])


def job(status, days=0.0, **fields) -> str:
    job_id = uuid.uuid4().hex
    record = {"id": job_id, "status": status, "stage": "Failed" if status == "failed" else "Working",
              "progress": 0.08, "class_code": "PSYC 1300", "start": "2026-09-30", "end": "2026-10-05",
              "created_at": time.time() - days * DAY, "pid": 1234, **fields}
    folder = lessons.LESSONS_DIR / ".jobs"
    path = folder / f"{job_id}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    (folder / f"{job_id}.log").write_text("", encoding="utf-8")
    stamp = time.time() - days * DAY
    os.utime(path, (stamp, stamp))
    return job_id


def log(job_id: str, text: str):
    (lessons.LESSONS_DIR / ".jobs" / f"{job_id}.log").write_text(text, encoding="utf-8")


class TestSnapshotShape:
    def test_a_healthy_machine_has_every_section_and_no_problems(self):
        snap = health.snapshot()
        assert snap["problems"] == []
        for name in SECTIONS:
            assert "error" not in snap[name], name
        assert snap["claude"] == {"cli_found": True, "cli_path": "C:/fake/claude.exe", "version": "2.1.4 (Claude Code)",
                                  "api_key": "not set", "last_test": None}
        assert snap["memory"] == {"available_gb": 8.0, "total_gb": 15.0, "level": "ok"}
        assert snap["disk"] == {"free_gb": 200.0, "total_gb": 500.0, "level": "ok"}
        assert snap["recorder"] == {"active": False, "class_code": None}
        assert snap["lesson_build"] == {"running": None, "last_failure": None}
        assert snap["narration"] == {"available": True, "reason": ""}
        assert snap["lid_watch"] == {"available": True}
        assert snap["llama_server"]["listening"] is False
        assert set(snap["storage"]) == {"notes_mb", "lessons_mb", "state_gb", "approximate"}

    def test_it_is_json_ready(self):
        json.dumps(health.snapshot())

    def test_polling_never_reaches_the_model(self):
        for _ in range(3):
            health.snapshot()
        assert world.model_calls == []

    def test_only_the_existence_of_the_api_key_is_reported(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-value")
        snap = health.snapshot()
        assert snap["claude"]["api_key"] == "set"
        assert "super-secret" not in json.dumps(snap)

    def test_an_empty_api_key_counts_as_not_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert health.snapshot()["claude"]["api_key"] == "not set"


class TestEverySectionFailsSoft:
    @pytest.mark.parametrize("name", SECTIONS)
    def test_one_broken_probe_never_hides_the_rest(self, name, monkeypatch):
        def boom():
            raise RuntimeError("probe exploded")

        monkeypatch.setitem(health._SECTIONS, name, boom)
        snap = health.snapshot()
        assert snap[name] == {"error": "RuntimeError: probe exploded"}
        for other in SECTIONS:
            if other != name:
                assert "error" not in snap[other], other
        assert any(p.startswith("Couldn't check") and "probe exploded" in p for p in snap["problems"])

    def test_real_failures_inside_the_probes_are_caught_too(self, monkeypatch):
        def broken(*args, **kwargs):
            raise OSError("no access")

        monkeypatch.setattr(health, "_read_memory", broken)
        monkeypatch.setattr(notes, "_find_claude_cli", broken)
        monkeypatch.setattr(health.shutil, "disk_usage", broken)
        monkeypatch.setattr(telemetry, "read_status", broken)
        monkeypatch.setattr(speech, "available", broken)
        monkeypatch.setattr(lesson_cleanup, "load_jobs", broken)
        snap = health.snapshot()
        for name in ("claude", "memory", "disk", "recorder", "lesson_build"):
            assert snap[name]["error"].startswith("OSError"), name
        assert snap["narration"]["available"] is False and "couldn't check" in snap["narration"]["reason"]
        assert "error" not in snap["storage"] and "error" not in snap["llama_server"]

    def test_a_failing_problem_summary_does_not_lose_the_sections(self, monkeypatch):
        monkeypatch.setattr(health, "_problems", lambda sections: 1 / 0)
        snap = health.snapshot()
        assert snap["problems"][0].startswith("Couldn't work out what is wrong")
        assert snap["memory"]["level"] == "ok"


class TestMemory:
    @pytest.mark.parametrize("available_gb, level", [
        (8, "ok"), (2.0, "ok"), (1.99, "low"), (1.5, "low"), (1.0, "low"), (0.99, "critical"), (0.1, "critical"), (0, "critical"),
    ])
    def test_levels(self, monkeypatch, available_gb, level):
        monkeypatch.setattr(health, "_read_memory", lambda: (int(available_gb * GIB), 16 * GIB))
        assert health.snapshot()["memory"]["level"] == level

    def test_available_ram_is_what_is_reported(self, monkeypatch):
        monkeypatch.setattr(health, "_read_memory", lambda: (int(6.25 * GIB), int(15.4 * GIB)))
        assert health.snapshot()["memory"] == {"available_gb": 6.25, "total_gb": 15.4, "level": "ok"}

    def test_critical_ram_warns_not_to_build_a_lesson(self, monkeypatch):
        monkeypatch.setattr(health, "_read_memory", lambda: (GIB // 2, 16 * GIB))
        assert health.snapshot()["problems"] == ["Less than 1 GB of RAM free - do not start a lesson build"]

    def test_low_ram_says_how_much_is_left(self, monkeypatch):
        monkeypatch.setattr(health, "_read_memory", lambda: (int(1.6 * GIB), 16 * GIB))
        [problem] = health.snapshot()["problems"]
        assert problem.startswith("Only 1.6 GB of RAM is free")

    def test_the_real_reading_is_physical_memory_in_plausible_range(self, monkeypatch):
        monkeypatch.undo()                  # this one test asks Windows for real
        if sys.platform != "win32":
            pytest.skip("Windows only")
        available, total = health._read_memory()
        assert 0 < available <= total and total > GIB

    @pytest.mark.parametrize("free_gb, level", [(200, "ok"), (5, "ok"), (4.9, "low"), (1, "low"), (0.5, "critical")])
    def test_disk_levels(self, monkeypatch, free_gb, level):
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(health.shutil, "disk_usage", lambda path: usage(500 * GIB, 0, int(free_gb * GIB)))
        assert health.snapshot()["disk"]["level"] == level

    def test_disk_is_the_drive_holding_the_project(self, monkeypatch):
        seen = []
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(health.shutil, "disk_usage", lambda path: seen.append(path) or usage(1, 1, 1))
        health.snapshot()
        assert seen == [health.ROOT]


class TestClaudeSection:
    def test_no_cli_is_reported_and_explained(self, monkeypatch):
        monkeypatch.setattr(notes, "_find_claude_cli", lambda: None)
        snap = health.snapshot()
        assert snap["claude"]["cli_found"] is False and snap["claude"]["version"] is None
        [problem] = snap["problems"]
        assert problem.startswith("Claude CLI not found") and "no ANTHROPIC_API_KEY" in problem

    def test_no_cli_but_a_key_is_milder(self, monkeypatch):
        monkeypatch.setattr(notes, "_find_claude_cli", lambda: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert health.snapshot()["problems"] == ["Claude CLI not found - the API key will be used instead"]

    def test_a_failed_test_with_no_cli_is_not_reported_as_a_second_problem(self, monkeypatch):
        monkeypatch.setattr(notes, "_find_claude_cli", lambda: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setattr(notes, "_try_claude_cli_format", lambda prompt, timeout=0: None)
        monkeypatch.setattr(notes, "last_cli_failure", "the Claude CLI isn't installed (claude.exe was not found)")
        health.test_claude({})
        assert health.snapshot()["problems"] == ["Claude CLI not found - the API key will be used instead"]

    def test_version_is_asked_once_a_minute_with_a_short_timeout_and_no_window(self, monkeypatch):
        clock = Clock()
        monkeypatch.setattr(health, "_clock", clock)
        calls = []

        def run(command, **kw):
            calls.append((command, kw))
            return subprocess.CompletedProcess(command, 0, "2.1.4 (Claude Code)\n", "")

        monkeypatch.setattr(health.subprocess, "run", run)
        for _ in range(4):
            health.snapshot()
        assert len(calls) == 1
        command, kw = calls[0]
        assert command == ["C:/fake/claude.exe", "--version"]
        assert kw["timeout"] == 5 and kw["stdin"] == subprocess.DEVNULL and kw["capture_output"] is True
        assert kw["creationflags"] == health._NO_WINDOW
        clock.advance(59)
        health.snapshot()
        assert len(calls) == 1
        clock.advance(2)
        health.snapshot()
        assert len(calls) == 2

    @pytest.mark.parametrize("outcome", ["timeout", "crash", "failed", "silent"])
    def test_a_version_probe_that_fails_costs_one_wait_a_minute_and_is_not_an_error(self, monkeypatch, outcome):
        calls = []

        def run(command, **kw):
            calls.append(command)
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(command, 5)
            if outcome == "crash":
                raise OSError("not a valid Win32 application")
            return subprocess.CompletedProcess(command, 1 if outcome == "failed" else 0, "" if outcome == "silent" else "oops", "")

        monkeypatch.setattr(health.subprocess, "run", run)
        first, second = health.snapshot(), health.snapshot()
        assert first["claude"]["version"] is None and "error" not in first["claude"] and first["claude"]["cli_found"]
        assert len(calls) == 1 and second["claude"]["version"] is None

    def test_a_different_cli_path_is_probed_again(self, monkeypatch):
        calls = []
        monkeypatch.setattr(health.subprocess, "run",
                            lambda command, **kw: calls.append(command) or subprocess.CompletedProcess(command, 0, "1.0\n", ""))
        health.snapshot()
        monkeypatch.setattr(notes, "_find_claude_cli", lambda: "C:/other/claude.exe")
        health.snapshot()
        assert [c[0] for c in calls] == ["C:/fake/claude.exe", "C:/other/claude.exe"]

    def test_polls_that_overlap_share_one_probe(self, monkeypatch):
        started, release, calls = threading.Event(), threading.Event(), []

        def run(command, **kw):
            calls.append(command)
            started.set()
            release.wait(5)
            return subprocess.CompletedProcess(command, 0, "2.1.4\n", "")

        monkeypatch.setattr(health.subprocess, "run", run)
        results = []
        first = threading.Thread(target=lambda: results.append(health.snapshot()))
        first.start()
        assert started.wait(2)
        second = threading.Thread(target=lambda: results.append(health.snapshot()))
        second.start()
        time.sleep(0.2)
        release.set()
        first.join(5)
        second.join(5)
        assert len(calls) == 1 and len(results) == 2


class TestTestClaude:
    @pytest.fixture
    def model(self, monkeypatch):
        """A fake CLI: replies are taken from model.replies in order; model.calls records each call."""
        state = types.SimpleNamespace(calls=[], replies=["OK"], failure=None, takes=0.0, clock=Clock())
        monkeypatch.setattr(health, "_clock", state.clock)
        monkeypatch.setattr(health, "_now", lambda: 5000.0 + state.clock.now)

        def ask(prompt, timeout=180):
            state.calls.append((prompt, timeout))
            state.clock.advance(state.takes)
            reply = state.replies.pop(0) if state.replies else None
            notes.last_cli_failure = None if reply else state.failure
            return reply

        monkeypatch.setattr(notes, "_try_claude_cli_format", ask)
        return state

    def test_one_small_real_call_and_the_result_is_kept(self, model):
        model.takes = 4.1
        result = health.test_claude({})
        assert result == {"ok": True, "seconds": 4.1, "reason": None, "at": 5000.0 + 1004.1}
        assert model.calls == [("Reply with exactly: OK", 45)]
        assert health.snapshot()["claude"]["last_test"] == result

    def test_the_result_is_written_atomically_and_survives_a_restart(self, model):
        result = health.test_claude({})
        saved = health.STATE_DIR / "health_claude.json"
        assert json.loads(saved.read_text(encoding="utf-8")) == result
        assert not list(health.STATE_DIR.glob("*.tmp"))
        health._last_result, health._result_loaded = None, False        # as after a restart
        assert health.snapshot()["claude"]["last_test"] == result

    def test_a_failure_carries_the_clis_own_reason(self, model):
        model.replies, model.failure = [], "the Claude CLI exited with code 1: Not logged in"
        result = health.test_claude({})
        assert result["ok"] is False and result["reason"] == "the Claude CLI exited with code 1: Not logged in"
        [problem] = health.snapshot()["problems"]
        assert problem == "The last Claude test failed: the Claude CLI exited with code 1: Not logged in"

    def test_a_failure_with_no_recorded_reason_still_says_something(self, model):
        model.replies, model.failure = [], None
        assert health.test_claude({})["reason"] == "the Claude CLI gave no reply"

    def test_a_45_second_timeout_is_not_called_zero_minutes(self, model):
        model.replies, model.failure = [], "the Claude CLI did not finish within 0 minutes"
        assert health.test_claude({})["reason"] == "the Claude CLI did not finish within 45 seconds"

    def test_an_exception_from_the_call_is_a_failed_test_not_an_error(self, monkeypatch, model):
        def explode(prompt, timeout=0):
            raise PermissionError("blocked")

        monkeypatch.setattr(notes, "_try_claude_cli_format", explode)
        result = health.test_claude({})
        assert result["ok"] is False and "PermissionError" in result["reason"]

    def test_a_passing_test_after_a_failed_one_clears_the_problem(self, model):
        model.replies, model.failure = [], "the Claude CLI isn't installed"
        health.test_claude({})
        assert health.snapshot()["problems"]
        model.clock.advance(11)
        model.replies = ["OK"]
        health.test_claude({})
        assert health.snapshot()["problems"] == []

    def test_an_old_failed_test_is_no_longer_a_current_problem(self, model, monkeypatch):
        model.replies, model.failure = [], "the Claude CLI isn't installed"
        health.test_claude({})
        model.clock.advance(2 * DAY)
        assert health.snapshot()["problems"] == []
        assert health.snapshot()["claude"]["last_test"]["ok"] is False      # still shown, just not a warning

    def test_a_repeat_within_ten_seconds_is_a_409_and_does_not_call_claude(self, model):
        health.test_claude({})
        model.clock.advance(5)
        with pytest.raises(health.Busy) as caught:
            health.test_claude({})
        assert caught.value.http_status == 409 and len(model.calls) == 1
        model.clock.advance(5.1)
        model.replies = ["OK"]
        health.test_claude({})
        assert len(model.calls) == 2

    def test_the_gap_is_counted_from_the_start_of_the_last_test(self, model):
        model.takes = 12                    # a slow test: by the time it ends, ten seconds are already up
        health.test_claude({})
        model.replies = ["OK"]
        health.test_claude({})
        assert len(model.calls) == 2

    def test_a_rejected_repeat_does_not_extend_the_wait(self, model):
        health.test_claude({})
        model.clock.advance(6)
        with pytest.raises(health.Busy):
            health.test_claude({})
        model.clock.advance(5)              # 11 s after the real test, only 5 after the rejected one
        model.replies = ["OK"]
        health.test_claude({})

    def test_two_tests_at_once_are_not_allowed(self, monkeypatch):
        inside, release, calls = threading.Event(), threading.Event(), []

        def slow(prompt, timeout=0):
            calls.append(prompt)
            inside.set()
            release.wait(5)
            return "OK"

        monkeypatch.setattr(notes, "_try_claude_cli_format", slow)
        first = []
        thread = threading.Thread(target=lambda: first.append(health.test_claude({})))
        thread.start()
        assert inside.wait(2)
        with pytest.raises(health.Busy) as caught:
            health.test_claude({})
        assert caught.value.http_status == 409 and "already being tested" in str(caught.value)
        release.set()
        thread.join(5)
        assert len(calls) == 1 and first[0]["ok"] is True

    def test_the_lock_is_released_after_a_failure(self, monkeypatch, model):
        model.replies, model.failure = [], "nope"
        health.test_claude({})
        model.clock.advance(11)
        model.replies = ["OK"]
        assert health.test_claude({})["ok"] is True

    @pytest.mark.parametrize("junk", ["{not json", "[]", '{"ok": "yes", "seconds": 1, "at": 1}',
                                      '{"ok": true, "seconds": "fast", "at": 1}', '{"ok": true, "seconds": 1}'])
    def test_a_damaged_result_file_is_ignored(self, junk):
        (health.STATE_DIR / "health_claude.json").write_text(junk, encoding="utf-8")
        assert health.snapshot()["claude"]["last_test"] is None

    def test_an_unwritable_state_folder_does_not_lose_the_result(self, model, monkeypatch):
        monkeypatch.setattr(health, "STATE_DIR", health.STATE_DIR / "missing" / "deeper" / "x")
        monkeypatch.setattr(health.Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(OSError("read-only")))
        result = health.test_claude({})
        assert result["ok"] is True and health.snapshot()["claude"]["last_test"] == result


class TestLessonBuildSection:
    def test_a_running_build_is_summarised(self):
        running = job("running", 0, stage="Writing the lesson with Claude", progress=0.08)
        world.alive.add(running)
        summary = health.snapshot()["lesson_build"]["running"]
        assert summary["id"] == running and summary["class_code"] == "PSYC 1300"
        assert summary["range"] == "2026-09-30 to 2026-10-05" and summary["stage"] == "Writing the lesson with Claude"
        assert summary["progress"] == 0.08

    def test_the_failure_reason_comes_from_the_jobs_error(self):
        job("failed", 1, error="The lesson could not be written: the Claude CLI did not finish within 60 minutes.")
        failure = health.snapshot()["lesson_build"]["last_failure"]
        assert failure["reason"] == "The lesson could not be written: the Claude CLI did not finish within 60 minutes."
        assert failure["class_code"] == "PSYC 1300" and failure["range"] == "2026-09-30 to 2026-10-05"
        assert failure["step"] == "Writing the lesson" and failure["status"] == "failed" and failure["superseded"] is False
        assert abs(failure["when"] - (time.time() - DAY)) < 60

    def test_with_no_error_it_falls_back_to_the_end_of_the_log(self):
        failed = job("failed", 1)
        log(failed, "Traceback (most recent call last):\n  File \"lessons.py\", line 1\n\nRuntimeError: Kokoro crashed\n\n")
        assert health.snapshot()["lesson_build"]["last_failure"]["reason"] == "RuntimeError: Kokoro crashed"

    def test_a_huge_log_is_read_only_at_its_end(self):
        failed = job("failed", 1)
        log(failed, "noise\n" * 200_000 + "the real cause\n")
        assert health.snapshot()["lesson_build"]["last_failure"]["reason"] == "the real cause"

    def test_with_neither_it_still_says_something(self):
        job("failed", 1)
        assert health.snapshot()["lesson_build"]["last_failure"]["reason"] == "No reason was recorded"

    def test_a_builder_that_died_says_so_even_when_its_log_has_a_stray_line(self):
        dead = job("running", 1, stage="Writing the lesson with Claude", progress=0.08)
        log(dead, "Warning: something harmless\n")
        reason = health.snapshot()["lesson_build"]["last_failure"]["reason"]
        assert reason == "The lesson builder stopped unexpectedly (last log line: Warning: something harmless)"

    def test_a_cancelled_build_gives_its_reason_not_a_line_from_its_log(self):
        cancelled = job("cancelled", 1)
        log(cancelled, "Stopping the children\n")
        assert health.snapshot()["lesson_build"]["last_failure"]["reason"] == "Cancelled before it finished"

    def test_a_builder_that_died_without_reporting_is_a_failure_and_its_file_is_left_alone(self):
        dead = job("running", 1, stage="Recording narration (3/9)", progress=0.6)
        path = lessons.LESSONS_DIR / ".jobs" / f"{dead}.json"
        before = (path.read_bytes(), path.stat().st_mtime)
        failure = health.snapshot()["lesson_build"]["last_failure"]
        assert failure["reason"] == "The lesson builder stopped unexpectedly" and failure["step"] == "Recording narration (3/9)"
        assert health.snapshot()["lesson_build"]["running"] is None
        assert (path.read_bytes(), path.stat().st_mtime) == before          # lessons.get_job would have rewritten it

    @pytest.mark.parametrize("progress, step", [
        (0.01, "Reading the notes"), (0.02, "Reading the notes"), (0.08, "Writing the lesson"), (0.2, "Writing the lesson"),
        (0.25, "Checking the lesson length"), (0.3, "Checking the lesson length"), (0.35, "Finding pictures"),
        (0.45, "Choosing the best pictures"), (0.5, "Recording narration"), (0.9, "Recording narration"),
    ])
    def test_the_step_is_worked_out_from_how_far_the_build_got(self, progress, step):
        job("failed", 1, progress=progress)
        assert health.snapshot()["lesson_build"]["last_failure"]["step"] == step

    def test_a_recorded_stage_wins_over_a_guess(self):
        job("failed", 1, progress=0.08, stage="Finding pictures")
        assert health.snapshot()["lesson_build"]["last_failure"]["step"] == "Finding pictures"

    def test_the_last_step_the_build_recorded_is_the_step_it_stopped_at(self):
        steps = [{"name": "Writing the lesson with Claude", "started_at": 1, "finished_at": 2},
                 {"name": "Recording narration", "started_at": 2, "finished_at": None, "detail": "3 of 9 slides"}]
        job("failed", 1, progress=0.08, stage="Failed", steps=steps)
        assert health.snapshot()["lesson_build"]["last_failure"]["step"] == "Recording narration"

    def test_a_failure_without_progress_has_no_step(self):
        job("failed", 1, progress=None)
        failure = health.snapshot()["lesson_build"]["last_failure"]
        assert failure["step"] is None
        assert health.snapshot()["problems"][0].startswith("The last lesson build failed: ")

    def test_the_newest_failed_or_cancelled_record_is_used(self):
        job("failed", 5, error="older")
        job("cancelled", 1, error="")
        job("failed", 3, error="middle")
        failure = health.snapshot()["lesson_build"]["last_failure"]
        assert failure["status"] == "cancelled" and failure["reason"] == "Cancelled before it finished"

    def test_a_later_success_makes_the_failure_old_news(self):
        job("failed", 3, error="boom")
        job("ready", 1)
        snap = health.snapshot()
        assert snap["lesson_build"]["last_failure"]["superseded"] is True
        assert snap["problems"] == []

    def test_a_later_running_build_also_supersedes_it(self):
        job("failed", 3, error="boom")
        world.alive.add(job("running", 0))
        assert health.snapshot()["problems"] == []

    def test_a_later_failure_does_not(self):
        job("failed", 3, error="old")
        job("failed", 1, error="new")
        assert health.snapshot()["problems"] == ["The last lesson build failed at Writing the lesson: new"]

    def test_a_cancelled_build_is_shown_but_is_not_a_problem(self):
        job("cancelled", 1)
        snap = health.snapshot()
        assert snap["lesson_build"]["last_failure"]["status"] == "cancelled" and snap["problems"] == []

    def test_the_problem_names_the_step_and_the_reason(self):
        job("failed", 1, error="The lesson could not be written: the Claude CLI did not finish within 60 minutes.")
        assert health.snapshot()["problems"] == [
            "The last lesson build failed at Writing the lesson: The lesson could not be written: "
            "the Claude CLI did not finish within 60 minutes"]

    def test_a_missing_lessons_folder_is_just_no_builds(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lessons, "LESSONS_DIR", tmp_path / "nowhere")
        assert health.snapshot()["lesson_build"] == {"running": None, "last_failure": None}

    def test_a_record_removed_during_the_poll_does_not_break_the_section(self, monkeypatch):
        gone = job("failed", 1, error="boom")
        real_is_file = Path.is_file

        def vanish_after_the_check(path):
            if path.name == f"{gone}.json" and real_is_file(path):
                path.unlink()
                return True
            return real_is_file(path)

        monkeypatch.setattr(Path, "is_file", vanish_after_the_check)
        assert "error" not in health.snapshot()["lesson_build"]


class TestOtherProbes:
    def test_the_recorder_is_active_only_with_a_fresh_heartbeat(self, monkeypatch):
        monkeypatch.setattr(health, "_now", lambda: 1000.0)
        for status, expected in [
            ({"active": True, "class_code": "BIOL 1440", "updated_at": 990.0}, {"active": True, "class_code": "BIOL 1440"}),
            ({"active": True, "class_code": "BIOL 1440", "updated_at": 969.0}, {"active": False, "class_code": None}),
            ({"active": False, "class_code": "BIOL 1440", "updated_at": 999.0}, {"active": False, "class_code": None}),
            ({"active": True, "class_code": "BIOL 1440"}, {"active": False, "class_code": None}),
            ({"active": True, "updated_at": 999.0}, {"active": True, "class_code": None}),
            ({"active": True, "updated_at": "yesterday"}, {"active": False, "class_code": None}),
            ({}, {"active": False, "class_code": None}),
        ]:
            monkeypatch.setattr(telemetry, "read_status", lambda status=status: status)
            assert health.snapshot()["recorder"] == expected, status

    def test_the_local_model_server_is_only_looked_at(self, monkeypatch):
        seen = []

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __getattr__(self, name):
                raise AssertionError(f"touched the connection: {name}")

        monkeypatch.setattr(health.socket, "create_connection", lambda address, timeout=None: seen.append((address, timeout)) or Connection())
        assert health.snapshot()["llama_server"] == {"listening": True, "port": 10000}
        assert seen == [(("127.0.0.1", 10000), 0.3)]

    def test_a_closed_port_is_not_listening(self):
        assert health.snapshot()["llama_server"]["listening"] is False

    def test_narration_reports_why_it_is_off(self, monkeypatch):
        monkeypatch.setattr(speech, "available", lambda: (False, "voice model files not found"))
        snap = health.snapshot()
        assert snap["narration"] == {"available": False, "reason": "voice model files not found"}
        assert snap["problems"] == ["Lessons will have no narration: voice model files not found"]

    def test_narration_is_checked_once_every_few_minutes(self, monkeypatch):
        clock = Clock()
        monkeypatch.setattr(health, "_clock", clock)
        calls = []
        monkeypatch.setattr(speech, "available", lambda: calls.append(1) or (True, ""))
        for _ in range(3):
            health.snapshot()
        assert len(calls) == 1
        clock.advance(301)
        health.snapshot()
        assert len(calls) == 2

    def test_the_lid_watcher_is_tried_once_and_stopped_at_once(self):
        for _ in range(4):
            assert health.snapshot()["lid_watch"] == {"available": True}
        assert len(FakeLid.instances) == 1 and FakeLid.instances[0].stopped

    def test_a_pc_without_a_lid(self):
        FakeLid.result = False
        assert health.snapshot()["lid_watch"] == {"available": False}
        assert FakeLid.instances[0].stopped
        assert health.snapshot()["problems"] == []

    def test_a_lid_watcher_that_cannot_start_is_not_an_error(self, monkeypatch):
        class Broken(FakeLid):
            def start(self, timeout=3.0):
                raise OSError("no window")

        monkeypatch.setattr(lid_watch, "LidWatcher", Broken)
        assert health.snapshot()["lid_watch"] == {"available": False}
        assert FakeLid.instances[0].stopped


class TestStorage:
    def test_sizes_of_notes_lessons_and_recordings(self):
        (health.NOTES_DIR / "a.md").write_bytes(b"x" * 3 * 1024 ** 2)
        (health.LESSONS_DIR / "L").mkdir()
        (health.LESSONS_DIR / "L" / "slide.ogg").write_bytes(b"x" * 5 * 1024 ** 2)
        (health.STATE_DIR / "audio" / "deep").mkdir(parents=True)
        (health.STATE_DIR / "audio" / "deep" / "a.wav").write_bytes(b"x" * 2 * 1024 ** 2)
        assert health.snapshot()["storage"] == {"notes_mb": 3.0, "lessons_mb": 5.0, "state_gb": 0.0, "approximate": False}

    def test_state_is_reported_in_gigabytes(self, monkeypatch):
        sizes = {health.NOTES_DIR: 0, health.LESSONS_DIR: 0, health.STATE_DIR: int(17.5 * GIB)}
        monkeypatch.setattr(health, "_walk_bytes", lambda path, deadline: (sizes[path], True))
        assert health.snapshot()["storage"]["state_gb"] == 17.5

    def test_the_walk_has_a_time_budget_and_says_when_it_ran_out(self, monkeypatch):
        clock = Clock()
        monkeypatch.setattr(health, "_clock", clock)
        for n in range(1000):
            (health.STATE_DIR / f"f{n}.wav").write_bytes(b"x" * 10)
        (health.NOTES_DIR / "a.md").write_bytes(b"x" * 1024 ** 2)
        real_walk = health._walk_bytes

        def walk(path, deadline):
            assert deadline == clock.now + health.STORAGE_BUDGET_SECONDS - (0 if path != health.STATE_DIR else 0) or True
            return real_walk(path, deadline)

        class Ticking(Clock):
            def __call__(self):
                self.now += 0.01            # every look at the clock costs 10 ms
                return self.now

        monkeypatch.setattr(health, "_clock", Ticking())
        storage = health.snapshot()["storage"]
        assert storage["approximate"] is True
        assert storage["notes_mb"] == 1.0 and storage["lessons_mb"] == 0.0     # reached before the time ran out
        assert 0 <= storage["state_gb"] < 0.0001                                # partial

    def test_folders_not_reached_before_the_deadline_are_unknown_not_zero(self, monkeypatch):
        calls = []

        def walk(path, deadline):
            calls.append(path)
            return 100, path != health.NOTES_DIR       # the first folder runs out of time

        monkeypatch.setattr(health, "_walk_bytes", walk)
        storage = health.snapshot()["storage"]
        assert calls == [health.NOTES_DIR]
        assert storage["approximate"] is True and storage["lessons_mb"] is None and storage["state_gb"] is None

    def test_the_result_is_cached_for_a_minute(self, monkeypatch):
        clock = Clock()
        monkeypatch.setattr(health, "_clock", clock)
        calls = []
        real_walk = health._walk_bytes
        monkeypatch.setattr(health, "_walk_bytes", lambda path, deadline: calls.append(path) or real_walk(path, deadline))
        health.snapshot()
        health.snapshot()
        assert len(calls) == 3
        clock.advance(61)
        (health.NOTES_DIR / "new.md").write_bytes(b"x" * 1024 ** 2)
        assert health.snapshot()["storage"]["notes_mb"] == 1.0 and len(calls) == 6

    def test_links_are_not_followed(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "big.bin").write_bytes(b"x" * 4 * 1024 ** 2)
        try:
            import _winapi
            _winapi.CreateJunction(str(outside), str(health.NOTES_DIR / "linked"))
        except (ImportError, OSError, AttributeError) as error:
            pytest.skip(f"can't make a junction here: {error}")
        assert health.snapshot()["storage"]["notes_mb"] == 0.0

    def test_missing_folders_count_as_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(health, "NOTES_DIR", tmp_path / "no-notes")
        assert health.snapshot()["storage"]["notes_mb"] == 0.0


class TestProblems:
    def test_most_urgent_first(self, monkeypatch):
        monkeypatch.setattr(health, "_read_memory", lambda: (GIB // 4, 16 * GIB))
        monkeypatch.setattr(notes, "_find_claude_cli", lambda: None)
        monkeypatch.setattr(speech, "available", lambda: (False, "kokoro-onnx is not installed"))
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(health.shutil, "disk_usage", lambda path: usage(500 * GIB, 0, GIB // 2))
        job("failed", 1, error="boom")
        problems = health.snapshot()["problems"]
        assert [p.split(" - ")[0].split(":")[0] for p in problems] == [
            "Less than 1 GB of RAM free", "Claude CLI not found", "The last lesson build failed at Writing the lesson",
            "Less than 1 GB of disk space is free", "Lessons will have no narration"]

    def test_every_problem_is_a_plain_sentence(self, monkeypatch):
        monkeypatch.setattr(health, "_read_memory", lambda: (GIB, 16 * GIB))
        job("failed", 1, error="boom")
        for problem in health.snapshot()["problems"]:
            assert isinstance(problem, str) and problem[0].isupper() and "{" not in problem and "None" not in problem


class TestThroughTheDashboard:
    """The real routes with the real health module (everything it reaches is still faked)."""

    @pytest.fixture
    def port(self, monkeypatch):
        monkeypatch.setattr(dashboard.Handler, "log_message", lambda *a, **k: None)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield httpd.server_port
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    def call(self, port, method, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request(method, path, "{}" if method == "POST" else None, headers or {})
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
        return response.status, body

    OWN = {"Content-Type": "application/json", "X-Notes-Dashboard": "1"}

    def test_the_snapshot_is_served_as_json(self, port):
        status, body = self.call(port, "GET", "/api/health")
        assert status == 200 and body["problems"] == [] and body["memory"]["level"] == "ok"

    def test_testing_claude_works_once_then_is_a_409(self, port, monkeypatch):
        monkeypatch.setattr(notes, "_try_claude_cli_format", lambda prompt, timeout=0: "OK")
        status, body = self.call(port, "POST", "/api/health/test-claude", self.OWN)
        assert status == 200 and body["ok"] is True
        status, body = self.call(port, "POST", "/api/health/test-claude", self.OWN)
        assert status == 409 and "wait" in body["error"]

    def test_testing_claude_needs_the_dashboards_own_page(self, port):
        status, _ = self.call(port, "POST", "/api/health/test-claude", {"Content-Type": "application/json"})
        assert status == 403
