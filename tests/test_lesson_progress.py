"""Lesson build progress: the step timeline, the time-left estimate, cancel and retry.

Claude, the voice and the image sources are faked and everything is written under tmp_path. The
only real processes are harmless sleeping Python children these tests start themselves - a
cancel is exercised against those, never against a real builder.
"""
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psutil
import pytest

import dashboard
import lessons
import speech
from test_lessons import NOTES, PLAN, fakes, library  # noqa: F401  (fixtures shared with the build tests)

JOB_ID = "a" * 32
BUILD = {"class_code": "CHEM 1450", "start": "2026-09-09", "end": "2026-09-09"}


def write_job(library, job_id=JOB_ID, **fields) -> Path:
    """A status file in lessons/.jobs, as start_lesson would leave it."""
    jobs = lessons.LESSONS_DIR / ".jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job = {"id": job_id, "status": "running", "stage": "Starting", "progress": 0.01, "voice": "af_heart",
           "focus": "", "minutes": 0, "created_at": time.time(), "lesson_id": None, "pid": None} | BUILD | fields
    path = jobs / f"{job_id}.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def step(name, started=1000.0, finished=None, **extra):
    return {"name": name, "started_at": started, "finished_at": finished, "detail": ""} | extra


def job_at(name, started=1000.0, **fields):
    """A running job whose current step is `name`, begun at t=1000."""
    return {"status": "running", "minutes": 0, "steps": [step(name, started)]} | fields


class TestTimeline:
    def run_builder(self, library, **fields):
        status = library / "job.json"
        status.write_text(json.dumps({"id": JOB_ID, "status": "running", "voice": "af_heart", "created_at": 0}
                                     | BUILD | fields), encoding="utf-8")
        return status, lessons.run_job(status)

    def test_a_build_records_each_step_once_in_order(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {"4": 2}})]
        status, code = self.run_builder(library)
        job = read(status)
        steps = job["steps"]
        assert code == 0 and job["status"] == "ready"
        assert [s["name"] for s in steps] == [lessons.STEP_READ, lessons.STEP_PLAN, lessons.STEP_FIND,
                                              lessons.STEP_PICK, lessons.STEP_NARRATE, lessons.STEP_FINISH]
        assert all(s["finished_at"] is not None and s["finished_at"] >= s["started_at"] for s in steps)
        assert all(a["finished_at"] <= b["started_at"] for a, b in zip(steps, steps[1:]))   # one step at a time
        assert job["stage"] == "Lesson ready" and job["progress"] == 1.0   # the older fields still work

    def test_narration_is_one_step_whose_detail_advances(self, library, fakes, monkeypatch):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        status = library / "job.json"
        seen = []

        def tts(text, path, voice=speech.DEFAULT_VOICE, speed=1.0):
            current = read(status)["steps"][-1]
            seen.append((current["name"], current["detail"], current["done"], current["total"]))
            path.write_bytes(b"OggS")
            return 10.0

        monkeypatch.setattr(lessons.speech, "synthesize", tts)
        status, _ = self.run_builder(library)
        assert seen == [(lessons.STEP_NARRATE, f"{n} of 5 slides", n, 5) for n in range(5)]
        narration = [s for s in read(status)["steps"] if s["name"] == lessons.STEP_NARRATE]
        assert len(narration) == 1 and narration[0]["detail"] == "5 of 5 slides"
        assert narration[0]["updated_at"] >= narration[0]["started_at"]

    def test_the_plan_size_is_recorded_for_the_estimate(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        status, _ = self.run_builder(library)
        job = read(status)
        assert job["slides"] == 5 and job["images"] == 1 and job["lesson_minutes"] > 0
        assert job["partial"].endswith(".partial") and job["partial"].startswith("CHEM_1450_20260909")

    def test_a_requested_length_adds_its_own_step(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), "not json - the rewrite is skipped", json.dumps({"choices": {}})]
        status, code = self.run_builder(library, minutes=1)
        names = [s["name"] for s in read(status)["steps"]]
        assert code == 0 and names[1:4] == [lessons.STEP_PLAN, lessons.STEP_LENGTH, lessons.STEP_FIND]

    def test_a_picker_with_nothing_to_choose_from_is_not_a_step(self, library, fakes, monkeypatch):
        """The search found no candidates, so Claude is never asked to pick: no pick step, and none promised."""
        fakes.replies = [json.dumps(PLAN)]
        monkeypatch.setattr(lessons, "image_candidates", lambda visual: [])
        status, code = self.run_builder(library)
        job = read(status)
        assert code == 0 and job["picker_requests"] == 0
        assert lessons.STEP_PICK not in [s["name"] for s in job["steps"]]
        upcoming = lessons.upcoming_steps(job_at(lessons.STEP_FIND, images=1, picker_requests=0, lesson_minutes=2))
        assert [s["name"] for s in upcoming] == [lessons.STEP_NARRATE, lessons.STEP_FINISH]

    def test_a_reused_plan_is_one_step_marked_as_reused(self, library, fakes, monkeypatch):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        available = lessons.speech.available
        monkeypatch.setattr(lessons.speech, "available", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))
        assert self.run_builder(library)[1] == 1
        monkeypatch.setattr(lessons.speech, "available", available)

        fakes.replies = [json.dumps({"choices": {}})]
        status, code = self.run_builder(library)
        job = read(status)
        assert code == 0 and job["plan_reused"] is True
        plan = [s for s in job["steps"] if s["name"] == lessons.STEP_PLAN]
        assert len(plan) == 1 and plan[0]["detail"] == "Reused the saved plan"

    def test_a_failed_build_closes_the_step_it_stopped_in(self, library, fakes, monkeypatch):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        monkeypatch.setattr(lessons.speech, "available", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))
        status, code = self.run_builder(library)
        job = read(status)
        assert code == 1 and job["status"] == "failed" and job["error"] == "disk full"
        assert job["steps"][-1]["name"] == lessons.STEP_PICK and job["steps"][-1]["finished_at"] is not None


class TestEstimate:
    def test_narration_time_left_comes_from_the_slides_already_recorded(self):
        # 10 of 50 slides took 200 s: 20 s each, counted just as slide 11 began
        recording = step(lessons.STEP_NARRATE, 1000, done=10, total=50, updated_at=1200)
        assert lessons.narration_remaining(recording, 1200) == 800
        assert lessons.narration_remaining(recording, 1210) == 790   # slide 11 is already 10 s in
        assert lessons.narration_remaining(recording, 5000) == 0     # never negative

    def test_narration_waits_for_enough_slides_before_measuring(self):
        for done in (None, 0, 1):
            assert lessons.narration_remaining(
                step(lessons.STEP_NARRATE, 1000, done=done, total=50, updated_at=1100), 1100) is None
        assert lessons.narration_remaining(step(lessons.STEP_NARRATE, 1000), 1100) is None   # no counts at all

    def test_measured_narration_has_no_range(self):
        job = job_at(lessons.STEP_NARRATE, done=10, total=50, updated_at=1200, slides=50, lesson_minutes=30)
        job["steps"][-1] |= {"done": 10, "total": 50, "updated_at": 1200}
        assert lessons.estimate_remaining(job, 1210) == {"seconds": 790, "range": None}

    def test_nothing_is_said_when_the_length_is_unknown(self):
        assert lessons.estimate_remaining(job_at(lessons.STEP_READ), 1000) is None
        assert lessons.estimate_remaining(job_at(lessons.STEP_PLAN), 1100) is None   # no length asked: unknown
        assert lessons.estimate_remaining({"status": "running"}, 1100) is None        # an old job without steps
        assert lessons.estimate_remaining(job_at("Some other step", minutes=20), 1100) is None

    def test_a_requested_length_gives_a_rough_range_from_the_constants(self):
        plan = lessons._plan_range(20)
        narration = lessons._spread(20 * lessons.NARRATION_PER_LESSON_MINUTE)
        estimate = lessons.estimate_remaining(job_at(lessons.STEP_PLAN, minutes=20), 1060)   # 60 s into writing
        # what's left of the plan, a possible rewrite (as long as the plan), pictures (the picker is
        # optional while the plan is unwritten, so it only counts toward the longest case), narration
        low = (plan[0] - 60) + 0 + 0 + 0 + narration[0]
        high = (plan[1] - 60) + plan[1] + lessons.OVERRUN_FLOOR + lessons.PICK_SECONDS[1] + narration[1]
        assert estimate["range"] == [round(low), round(high)]
        assert estimate["seconds"] == round((low + high) / 2)

    def test_the_estimate_shrinks_as_the_step_goes_on_and_never_goes_negative(self):
        job = job_at(lessons.STEP_PLAN, minutes=20)
        early, later = (lessons.estimate_remaining(job, 1000 + s)["seconds"] for s in (30, 300))
        assert later < early
        overrun = lessons.estimate_remaining(job, 1000 + 3600)   # far past the usual time
        assert overrun["range"][0] >= 0 and overrun["range"][1] > 0

    def test_a_reused_plan_makes_the_length_known_even_without_a_request(self):
        job = job_at(lessons.STEP_PLAN, plan_reused=True, lesson_minutes=10, slides=30, images=4)
        narration = lessons._spread(10 * lessons.NARRATION_PER_LESSON_MINUTE)
        unsure = lessons.estimate_remaining(job, 1001)
        # 4 images may still find no candidates, so the picker only adds to the longest case until the find step ends
        assert narration[0] <= unsure["range"][0] < narration[0] + lessons.PICK_SECONDS[0]
        sure = lessons.estimate_remaining(job | {"picker_requests": 4}, 1001)
        assert sure["range"][0] >= narration[0] + lessons.PICK_SECONDS[0]   # candidates were found: the picker runs

    def test_a_lesson_without_real_pictures_skips_the_picker(self):
        job = job_at(lessons.STEP_FIND, lesson_minutes=10, slides=30, images=0)
        assert [s["name"] for s in lessons.upcoming_steps(job)] == [lessons.STEP_NARRATE, lessons.STEP_FINISH]

    def test_upcoming_steps_follow_the_current_one(self):
        job = job_at(lessons.STEP_PLAN, minutes=20)
        assert lessons.upcoming_steps(job) == [
            {"name": lessons.STEP_LENGTH, "optional": False}, {"name": lessons.STEP_FIND, "optional": False},
            {"name": lessons.STEP_PICK, "optional": True},   # unknown until the plan says how many images
            {"name": lessons.STEP_NARRATE, "optional": False}, {"name": lessons.STEP_FINISH, "optional": False}]
        assert lessons.upcoming_steps(job_at(lessons.STEP_NARRATE, images=2)) == [
            {"name": lessons.STEP_FINISH, "optional": False}]
        assert lessons.upcoming_steps({"status": "running"}) == []   # an old job

    def test_get_job_adds_the_estimate_without_saving_it(self, library, spawned):
        builder = live_builder(spawned)
        path = write_job(library, minutes=20, steps=[step(lessons.STEP_PLAN, time.time() - 60)], pid=builder)
        job = lessons.get_job(JOB_ID)
        assert job["eta_seconds"] > 0 and len(job["eta_range"]) == 2
        assert [s["name"] for s in job["upcoming"]][-1] == lessons.STEP_FINISH
        assert "eta_seconds" not in read(path)

    def test_get_job_copes_with_a_job_from_before_steps_existed(self, library, spawned):
        write_job(library, pid=live_builder(spawned))
        job = lessons.get_job(JOB_ID)
        assert job["eta_seconds"] is None and job["eta_range"] is None and job["upcoming"] == []

    def test_get_job_copes_with_a_malformed_steps_list(self, library, spawned):
        write_job(library, pid=live_builder(spawned), steps=[{"name": lessons.STEP_PLAN}])   # no started_at
        assert lessons.get_job(JOB_ID)["eta_seconds"] is None


class TestPlanReuse:
    @pytest.fixture
    def failed_build(self, library, fakes, monkeypatch):
        """A build that wrote its plan and then failed, like a lost voice file or a crash."""
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        monkeypatch.setattr(lessons.speech, "available", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))
        spawned = []
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: spawned.append(path) or None)
        return fakes, spawned, monkeypatch

    def start_and_run(self, spawned, payload):
        job_id = lessons.start_lesson(payload)["job_id"]
        return job_id, lessons.run_job(spawned[-1])

    def test_the_helper_names_the_file_the_build_writes(self, library, failed_build):
        _, spawned, _ = failed_build
        job_id, code = self.start_and_run(spawned, BUILD)
        _, notes_markdown, _ = lessons.notes_for_range("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9))
        saved = list((lessons.LESSONS_DIR / ".plans").glob("*.json"))
        assert code == 1 and saved == [lessons.plan_cache_path("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9),
                                                               notes_markdown)]

    def test_a_failed_job_says_its_plan_is_saved_and_a_ready_one_does_not(self, library, failed_build):
        _, spawned, _ = failed_build
        job_id, _ = self.start_and_run(spawned, BUILD)
        assert lessons.get_job(job_id)["plan_saved"] is True
        write_job(library, job_id="b" * 32, status="ready", lesson_id="x")
        assert lessons.get_job("b" * 32)["plan_saved"] is False

    def test_a_changed_request_or_changed_notes_or_an_old_plan_is_not_saved(self, library, failed_build):
        _, spawned, _ = failed_build
        self.start_and_run(spawned, BUILD | {"focus": "alcohols"})
        focused = write_job(library, job_id="c" * 32, status="failed", focus="alcohols")
        assert lessons.get_job("c" * 32)["plan_saved"] is True
        write_job(library, job_id="d" * 32, status="failed", focus="something else")
        assert lessons.get_job("d" * 32)["plan_saved"] is False       # a different focus is a different plan
        write_job(library, job_id="e" * 32, status="failed", focus="alcohols", minutes=5)
        assert lessons.get_job("e" * 32)["plan_saved"] is False       # and so is a different length

        notes = lessons.NOTES_DIR / "CHEM_1450.md"
        notes.write_text(notes.read_text(encoding="utf-8") + "- Alkanes are saturated\n", encoding="utf-8")
        assert lessons.get_job("c" * 32)["plan_saved"] is False       # the notes moved on since
        notes.write_text(NOTES, encoding="utf-8")
        assert lessons.get_job("c" * 32)["plan_saved"] is True

        plan = next((lessons.LESSONS_DIR / ".plans").glob("*.json"))
        old = time.time() - (lessons.PLAN_CACHE_DAYS + 1) * 86400
        os.utime(plan, (old, old))
        assert lessons.get_job("c" * 32)["plan_saved"] is False       # the same 7-day rule the build applies
        assert focused.exists()

    def test_a_running_build_says_whether_its_plan_is_saved_so_cancel_can_say_so(self, library):
        write_job(library)                                       # running, and its builder counts as alive
        assert lessons.get_job(JOB_ID)["plan_saved"] is False
        _, notes_markdown, _ = lessons.notes_for_range("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9))
        plan = lessons.plan_cache_path("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), notes_markdown)
        plan.parent.mkdir(parents=True)
        plan.write_text(json.dumps(PLAN), encoding="utf-8")      # written once the plan is ready
        assert lessons.get_job(JOB_ID)["plan_saved"] is True

    def test_a_job_for_a_class_that_is_gone_is_just_not_saved(self, library):
        write_job(library, status="failed", class_code="BIOL 9999")
        assert lessons.get_job(JOB_ID)["plan_saved"] is False

    def test_trying_again_with_the_jobs_own_settings_reuses_the_plan(self, library, failed_build):
        fakes, spawned, monkeypatch = failed_build
        fakes.replies = [json.dumps(PLAN), "not json - the length rewrite is skipped", json.dumps({"choices": {}})]
        payload = BUILD | {"focus": "alcohols", "minutes": "1"}
        job_id, code = self.start_and_run(spawned, payload)
        job = lessons.get_job(job_id)
        assert code == 1 and job["plan_saved"] is True

        # The voice works now - and the page's retry sends what the job holds. Only the voice is
        # re-patched: undoing everything would also drop the library fixture's temp folders and
        # send this build to the real lessons/ and notes/.
        monkeypatch.setattr(lessons.speech, "available", lambda: (True, ""))
        monkeypatch.setattr(lessons, "_ask_claude", lambda prompt, timeout: fakes.prompts.append(prompt) or
                            json.dumps({"choices": {}}))
        monkeypatch.setattr(lessons, "image_candidates", lambda visual: [])
        again = {"class_code": job["class_code"], "start": job["start"], "end": job["end"], "voice": job["voice"],
                 "focus": job["focus"], "minutes": str(job["minutes"]) if job["minutes"] else ""}
        before = len(fakes.prompts)
        new_id, code = self.start_and_run(spawned, again)
        assert code == 0 and read(spawned[-1])["lesson_id"]
        assert all("outstanding tutor" not in prompt for prompt in fakes.prompts[before:])   # no second plan
        assert not list((lessons.LESSONS_DIR / ".plans").glob("*.json"))                      # cleared once built


class TestCancel:
    @pytest.fixture
    def stopped(self, monkeypatch):
        """Replaces the process handling: records what would have been killed."""
        killed = []
        monkeypatch.setattr(lessons, "_builder_process", lambda job: "builder")
        monkeypatch.setattr(lessons, "_kill_process_tree", lambda proc: killed.append(proc))
        monkeypatch.setattr(lessons, "_builder_alive", lambda job: False)   # as it is once the builder is killed
        return killed

    def running_job(self, library, **fields):
        folder = lessons.LESSONS_DIR / "CHEM_1450_20260909-20260909_20261001000000.partial"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "slide-01.ogg").write_bytes(b"OggS")
        plans = lessons.LESSONS_DIR / ".plans"
        plans.mkdir(parents=True, exist_ok=True)
        (plans / "CHEM_1450_20260909-20260909_abc.json").write_text("{}", encoding="utf-8")
        return write_job(library, pid=4242, partial=folder.name,
                         steps=[step(lessons.STEP_READ, 1.0, 2.0), step(lessons.STEP_NARRATE, 2.0)], **fields), folder

    def test_cancel_stops_the_builder_and_marks_the_job_cancelled(self, library, stopped):
        path, folder = self.running_job(library)
        job = lessons.cancel_job(JOB_ID)
        assert stopped == ["builder"]
        assert job["status"] == "cancelled" and job["stage"] == "Cancelled" and job["message"]
        assert read(path)["status"] == "cancelled"
        assert job["steps"][-1]["name"] == lessons.STEP_NARRATE and job["steps"][-1]["finished_at"] is not None
        assert not folder.exists()                                              # the half-built lesson is gone...
        assert list((lessons.LESSONS_DIR / ".plans").glob("*.json"))            # ...the saved plan is kept

    def test_a_cancelled_job_is_not_later_called_a_crash_and_does_not_block_a_new_build(self, library, stopped,
                                                                                       monkeypatch):
        self.running_job(library)
        lessons.cancel_job(JOB_ID)
        assert lessons.get_job(JOB_ID)["status"] == "cancelled"                  # the builder is "dead", still cancelled
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: None)
        assert lessons.start_lesson(BUILD)["job_id"] != JOB_ID                   # no LessonBusy

    def test_a_poll_during_the_kill_cannot_turn_the_cancel_into_a_failure(self, library, monkeypatch):
        """The builder looks dead the moment it is killed; a poll then must wait for the cancel to finish."""
        self.running_job(library)
        polled = []
        monkeypatch.setattr(lessons, "_builder_process", lambda job: "builder")
        monkeypatch.setattr(lessons, "_builder_alive", lambda job: False)

        def kill(proc):
            # the poll sleeps 0.2 s, then waits for the lock this cancel holds; the 0.3 s here lets it get there
            poller = threading.Thread(target=lambda: polled.append(lessons.get_job(JOB_ID)["status"]))
            poller.start()
            time.sleep(0.3)
            kill.poller = poller

        monkeypatch.setattr(lessons, "_kill_process_tree", kill)
        lessons.cancel_job(JOB_ID)
        kill.poller.join(5)
        assert polled == ["cancelled"]

    def test_a_job_that_never_recorded_its_folder_deletes_no_folder_by_name_match(self, library, stopped):
        """A stale record can say 'running' and a same-dates folder can belong to a newer build, so
        cancel removes only the folder the job itself recorded."""
        path, folder = self.running_job(library)
        job = read(path)
        del job["partial"]
        path.write_text(json.dumps(job), encoding="utf-8")
        other = lessons.LESSONS_DIR / "CHEM_1450_20260909-20260909_20261001999999.partial"
        other.mkdir()
        lessons.cancel_job(JOB_ID)
        assert folder.exists() and other.exists()

    def test_a_build_that_finished_just_before_the_kill_is_left_alone(self, library, monkeypatch):
        path, folder = self.running_job(library)
        monkeypatch.setattr(lessons, "_builder_process", lambda job: "builder")
        monkeypatch.setattr(lessons, "_builder_alive", lambda job: True)

        def kill(proc):   # it wrote "ready" in the instant before it was stopped
            path.write_text(json.dumps(read(path) | {"status": "ready", "lesson_id": "CHEM_x"}), encoding="utf-8")

        monkeypatch.setattr(lessons, "_kill_process_tree", kill)
        job = lessons.cancel_job(JOB_ID)
        assert job["status"] == "ready" and job["lesson_id"] == "CHEM_x"

    def test_a_builder_reporting_a_failure_while_it_is_killed_still_reads_as_cancelled(self, library, monkeypatch):
        """Killing the claude child first makes the builder write 'failed' (no plan, no reply) before its
        own kill lands. The person pressed Cancel, so that must not show as a crash."""
        path, folder = self.running_job(library)
        monkeypatch.setattr(lessons, "_builder_process", lambda job: "builder")
        monkeypatch.setattr(lessons, "_builder_alive", lambda job: True)

        def kill(proc):
            path.write_text(json.dumps(read(path) | {"status": "failed", "stage": "Failed",
                                                     "error": "The lesson could not be written"}), encoding="utf-8")

        monkeypatch.setattr(lessons, "_kill_process_tree", kill)
        job = lessons.cancel_job(JOB_ID)
        assert job["status"] == "cancelled" and "error" not in job and job["message"]
        assert read(path)["status"] == "cancelled" and not folder.exists()

    def test_a_job_with_no_pid_from_long_ago_is_unstuck_without_a_kill(self, library, monkeypatch):
        """A builder that never reported a pid and is past its start-up window has nothing to kill."""
        write_job(library, created_at=time.time() - 600)
        monkeypatch.setattr(lessons, "_kill_process_tree", lambda proc: pytest.fail("nothing to kill"))
        assert lessons.cancel_job(JOB_ID)["status"] == "cancelled"

    def test_what_cannot_be_cancelled_is_refused_with_a_409_style_error(self, library, stopped):
        with pytest.raises(FileNotFoundError):
            lessons.cancel_job("f" * 32)
        with pytest.raises(FileNotFoundError):
            lessons.cancel_job("../../etc")
        for status, words in (("ready", "finished"), ("failed", "stopped"), ("cancelled", "cancelled")):
            write_job(library, status=status)
            with pytest.raises(lessons.CancelRefused, match=words) as refused:
                lessons.cancel_job(JOB_ID)
            assert refused.value.http_status == 409
        assert stopped == []                                    # nothing was killed
        assert not issubclass(lessons.CancelRefused, (ValueError, TypeError))   # or it would map to a 400

    def test_a_build_still_starting_cannot_be_identified_yet(self, library):
        write_job(library, pid=None)   # not in _processes and has not reported its pid
        with pytest.raises(lessons.CancelRefused, match="still starting"):
            lessons.cancel_job(JOB_ID)

    def test_the_request_needs_a_job_id(self, library):
        for payload in ({}, {"job_id": 7}, {"job_id": None}):
            with pytest.raises(ValueError):
                lessons.cancel_job_request(payload)
        write_job(library, status="failed")
        with pytest.raises(lessons.CancelRefused):
            lessons.cancel_job_request({"job_id": JOB_ID})

    def test_only_the_builders_command_line_counts(self):
        job = "a" * 32
        build = ["py", "C:\\app\\src\\lessons.py", "--job", f"C:\\app\\lessons\\.jobs\\{job}.json"]
        assert lessons._is_builder_command(build, job)
        assert lessons._is_builder_command(["py", "x.py", "--job", f"/tmp/{job}.json"], job)   # either slash style
        assert not lessons._is_builder_command(build, "b" * 32)                                 # another job's builder
        assert not lessons._is_builder_command(["py", "-c", "pass"], job)
        assert not lessons._is_builder_command(["py", "lessons.py", "--job"], job)
        assert not lessons._is_builder_command(["py", f"{job}.json"], job)                      # no --job


# --- real processes: harmless sleeping Python children started here -------------------------------

SLEEP = "import time; time.sleep(120)"
# A stand-in builder: records its own pid and a grandchild's, and sleeps - like a builder with a claude child
STAND_IN = ("import os, subprocess, sys, time; "
            f"kid = subprocess.Popen([sys.executable, '-c', '{SLEEP}']); "
            "open(sys.argv[1], 'w').write(f'{os.getpid()} {kid.pid}'); time.sleep(120)")


class Spawned:
    """Starts the test's own processes and, whatever happens, ends exactly those and their children."""

    def __init__(self):
        self.popens, self.family = [], set()

    def start(self, *command) -> subprocess.Popen:
        popen = subprocess.Popen([sys.executable, "-c", *command], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.popens.append(popen)
        return popen

    def remember(self, pid):
        """Notes a process and its descendants, so they can be checked and cleaned up."""
        try:
            parent = psutil.Process(pid)
            self.family |= {parent.pid} | {child.pid for child in parent.children(recursive=True)}
        except psutil.NoSuchProcess:
            pass

    def stop(self):
        for pid in self.family | {popen.pid for popen in self.popens}:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass


@pytest.fixture
def spawned():
    processes = Spawned()
    yield processes
    processes.stop()


def wait_for(condition, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False


def gone(pids):
    return all(not psutil.pid_exists(pid) for pid in pids)


def start_stand_in(spawned, library, job_id=JOB_ID):
    """A running 'builder' whose command line is `python -c ... --job <status file>`, plus its job file."""
    status = lessons.LESSONS_DIR / ".jobs" / f"{job_id}.json"
    pids = library / f"{job_id}.pids"
    process = spawned.start(STAND_IN, str(pids), "--job", str(status))
    assert wait_for(pids.exists) and wait_for(lambda: pids.read_text().count(" ") == 1)
    own, kid = (int(n) for n in pids.read_text().split())
    spawned.remember(process.pid)
    assert {own, kid} <= spawned.family and len(spawned.family) >= 3   # launcher layers and a grandchild
    return process, own


def live_builder(spawned, job_id=JOB_ID) -> int:
    """A sleeping process whose command line names the job's status file: it counts as that build's
    builder. A plain sleeping child does not, since a reused pid must never look like a running build."""
    status = lessons.LESSONS_DIR / ".jobs" / f"{job_id}.json"
    return spawned.start(SLEEP, "--job", str(status)).pid


class TestRealProcesses:
    def test_cancel_kills_the_builder_and_every_child_it_started(self, library, spawned):
        """A builder started by an earlier dashboard: only the pid in its status file is known."""
        (lessons.LESSONS_DIR / ".jobs").mkdir(parents=True)
        process, own = start_stand_in(spawned, library)
        write_job(library, pid=own, steps=[step(lessons.STEP_PLAN, time.time())])
        job = lessons.cancel_job(JOB_ID)
        assert job["status"] == "cancelled"
        assert wait_for(lambda: gone(spawned.family)), [p for p in spawned.family if psutil.pid_exists(p)]

    def test_cancel_of_a_build_this_dashboard_started_ends_the_whole_tree_too(self, library, spawned, monkeypatch):
        (lessons.LESSONS_DIR / ".jobs").mkdir(parents=True)
        process, _ = start_stand_in(spawned, library)
        monkeypatch.setitem(lessons._processes, JOB_ID, process)
        write_job(library, pid=None)   # the builder never got to report its pid
        assert lessons.cancel_job(JOB_ID)["status"] == "cancelled"
        assert wait_for(lambda: gone(spawned.family)) and process.poll() is not None
        assert JOB_ID not in lessons._processes

    def test_the_builder_is_recognized_by_its_command_line(self, library, spawned):
        (lessons.LESSONS_DIR / ".jobs").mkdir(parents=True)
        process, own = start_stand_in(spawned, library)
        job = {"id": JOB_ID, "pid": own}
        found = lessons._builder_process(job)
        assert found is not None and found.pid == own
        assert lessons._builder_process({"id": "b" * 32, "pid": own}) is None   # another job's pid: not its builder

    def test_a_reused_pid_is_never_killed(self, library, spawned):
        """The pid in the job file now belongs to an unrelated program: the builder is long dead."""
        bystander = spawned.start(SLEEP)
        spawned.remember(bystander.pid)
        wait_for(lambda: len(psutil.Process(bystander.pid).cmdline()) > 1)
        write_job(library, pid=bystander.pid, steps=[step(lessons.STEP_PLAN, time.time())])
        job = lessons.cancel_job(JOB_ID)                   # it just unsticks the job...
        assert job["status"] == "cancelled"
        assert bystander.poll() is None and psutil.pid_exists(bystander.pid)   # ...and kills nothing

    def test_our_own_child_that_does_not_look_like_a_builder_is_refused_and_left_running(self, library, spawned,
                                                                                         monkeypatch):
        child = spawned.start(SLEEP)
        spawned.remember(child.pid)
        monkeypatch.setitem(lessons._processes, JOB_ID, child)
        write_job(library, pid=None)
        with pytest.raises(lessons.CancelRefused, match="nothing was stopped"):
            lessons.cancel_job(JOB_ID)
        assert child.poll() is None
        assert read(lessons.LESSONS_DIR / ".jobs" / f"{JOB_ID}.json")["status"] == "running"

    def test_a_pid_that_no_longer_exists_just_unsticks_the_job(self, library):
        quick = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        quick.wait()
        write_job(library, pid=quick.pid)
        assert lessons.cancel_job(JOB_ID)["status"] == "cancelled"   # cancelled, not a crash

    def test_a_reused_pid_is_not_taken_for_a_running_build(self, library, spawned, monkeypatch):
        """Once the builder has gone its pid can belong to any program. The build must read as failed, not
        running: a running record would block every new build with 'already being built'."""
        bystander = spawned.start(SLEEP)
        spawned.remember(bystander.pid)
        wait_for(lambda: len(psutil.Process(bystander.pid).cmdline()) > 1)
        write_job(library, pid=bystander.pid, steps=[step(lessons.STEP_PLAN, time.time())])
        assert lessons.get_job(JOB_ID)["status"] == "failed"
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: None)   # the new build itself is not started here
        assert lessons.start_lesson(BUILD)["job_id"] != JOB_ID
        assert bystander.poll() is None                                    # and the bystander is left alone

    def test_a_builder_whose_command_cannot_be_read_is_not_counted_as_running(self, monkeypatch):
        def unreadable(pid):
            raise psutil.AccessDenied(pid)
        monkeypatch.setattr(lessons.psutil, "Process", unreadable)
        assert lessons._builder_alive({"id": JOB_ID, "pid": 4242, "created_at": time.time()}) is False

    @pytest.mark.skipif(os.name != "nt", reason="taskkill is Windows only")
    def test_a_taskkill_that_hangs_is_reported_as_a_refusal_not_a_crash(self, monkeypatch):
        real_run = subprocess.run

        def hung(command, **kwargs):
            if command[0] == "taskkill":
                raise subprocess.TimeoutExpired(command, 30)
            return real_run(command, **kwargs)   # nothing else is run here; no real pid is ever named

        monkeypatch.setattr(lessons.subprocess, "run", hung)

        class Builder:
            pid = 424242

        with pytest.raises(lessons.CancelRefused, match="would not stop"):
            lessons._kill_process_tree(Builder())

    def test_the_fallback_tree_kill_also_takes_the_whole_tree(self, library, spawned):
        """What other platforms use in place of taskkill: children first, then the process."""
        (lessons.LESSONS_DIR / ".jobs").mkdir(parents=True)
        process, own = start_stand_in(spawned, library)
        proc = psutil.Process(own)
        lessons._kill_process_tree_with_psutil(proc)
        assert wait_for(lambda: gone({own} | {c for c in spawned.family if c != process.pid}))


class TestRoute:
    @pytest.fixture
    def server(self, monkeypatch):
        monkeypatch.setattr(dashboard.Handler, "log_message", lambda *a, **k: None)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield httpd.server_port
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    def post(self, port, body, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/api/lessons/cancel", json.dumps(body),
                     headers or {"Content-Type": "application/json", "X-Notes-Dashboard": "1"})
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def test_cancel_errors_reach_the_page_with_their_own_status_and_message(self, library, server):
        write_job(library, status="ready")
        status, data = self.post(server, {"job_id": JOB_ID})
        assert status == 409 and "already finished" in data["error"]
        assert self.post(server, {"job_id": "0" * 32})[0] == 404
        assert self.post(server, {})[0] == 400
        assert self.post(server, {"job_id": JOB_ID}, {"Content-Type": "application/json"})[0] == 403   # not from the page

    def test_cancel_over_http_returns_the_cancelled_job(self, library, server, monkeypatch):
        write_job(library, pid=4242, steps=[step(lessons.STEP_PLAN, time.time())])
        monkeypatch.setattr(lessons, "_builder_process", lambda job: None)
        status, data = self.post(server, {"job_id": JOB_ID})
        assert status == 200 and data["status"] == "cancelled" and data["plan_saved"] is False
