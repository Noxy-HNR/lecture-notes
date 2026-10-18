"""Freestyle lessons: a lesson on whatever the person describes, with no notes behind it.

Claude, the voice and the image sources are faked (see test_lessons.fakes) and everything is
written under tmp_path.
"""
import http.client
import json
import sys
import threading
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import dashboard
import lessons
import speech
from test_lessons import NOTES, PLAN, fakes, library  # noqa: F401  (fixtures shared with the build tests)

TOPIC = "how noise-cancelling headphones work, from scratch"
TODAY = date.today()


def build(**kwargs):
    kwargs.setdefault("topic", TOPIC)
    return lessons.build_lesson(lessons.FREESTYLE_CODE, TODAY, TODAY, "af_heart", **kwargs)


class TestRequestText:
    def test_the_description_is_tidied(self):
        assert lessons.clean_topic("  how  vaccines\n\twork \x07 ") == "how vaccines work"

    @pytest.mark.parametrize("bad", [None, 5, "", "   ", "ab", ["x"]])
    def test_nothing_to_learn_is_refused(self, bad):
        with pytest.raises(ValueError, match="Describe what you want to learn"):
            lessons.clean_topic(bad)

    def test_a_novel_is_refused(self):
        lessons.clean_topic("x" * lessons.MAX_TOPIC_CHARS)
        with pytest.raises(ValueError, match="under"):
            lessons.clean_topic("x" * (lessons.MAX_TOPIC_CHARS + 1))

    def test_a_long_request_gets_a_short_title_on_a_word_boundary(self):
        assert lessons.topic_title("how vaccines work") == "how vaccines work"
        title = lessons.topic_title("explain the causes of the French Revolution " * 4)
        assert len(title) <= 81 and title.endswith("…") and not title.endswith(" …")


class TestPrompt:
    def test_the_prompt_is_about_the_request_and_has_no_notes(self):
        prompt = lessons.build_freestyle_prompt(TOPIC)
        flat = " ".join(prompt.split())
        assert f'"{TOPIC}"' in prompt
        assert "There are no class notes" in flat and "NOTES (" not in prompt and "found the lectures" not in prompt
        assert "Never invent facts" in flat and "uncertain, disputed or changes over time" in flat

    def test_it_teaches_and_answers_in_the_same_format_as_a_notes_lesson(self):
        free = lessons.build_freestyle_prompt(TOPIC, 12)
        notes = lessons.build_lesson_prompt("Chemistry", "CHEM 1450", "2026-09-09", "2026-09-09", "- a point", "", 12)
        teaching = lessons._teaching_sections(12)
        assert teaching in free and teaching in notes
        for needle in ("HOW TO TEACH", "VISUALS", '"slides": [', '"heads_up"', "about 12 minutes"):
            assert needle in free

    def test_no_length_means_the_model_decides(self):
        flat = " ".join(lessons.build_freestyle_prompt(TOPIC).split())
        assert "no slide limit and no time limit" in flat and "LENGTH:" not in flat

    def test_the_notes_prompt_is_unchanged_by_sharing_the_teaching_half(self):
        prompt = lessons.build_lesson_prompt("Chemistry", "CHEM 1450", "2026-09-09", "2026-09-09", "- a point")
        assert prompt.startswith("You are an outstanding tutor. A student in Chemistry (CHEM 1450)")
        assert prompt.endswith("NOTES (2026-09-09):\n---\n- a point\n---\n")
        assert "\n\n\n" not in prompt


class TestStarting:
    @pytest.fixture
    def spawned(self, library, monkeypatch):
        started = []
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: started.append(path) or None)
        return started

    def test_a_description_starts_a_job_with_no_class_notes_or_dates(self, spawned):
        lessons.start_lesson({"topic": "  how  do vaccines work\n", "minutes": "10"})
        job = json.loads(spawned[0].read_text(encoding="utf-8"))
        assert job["topic"] == "how do vaccines work" and job["kind"] == "freestyle"
        assert job["class_code"] == lessons.FREESTYLE_CODE and job["focus"] == ""
        assert job["start"] == job["end"] == TODAY.isoformat() and job["minutes"] == 10

    def test_a_blank_or_bad_description_is_refused_before_any_job_exists(self, spawned):
        for payload in ({"topic": ""}, {"topic": "  "}, {"topic": 7}):
            with pytest.raises(ValueError):
                lessons.start_lesson(payload)
        assert spawned == [] and not list((lessons.LESSONS_DIR / ".jobs").glob("*.json"))

    def test_a_class_and_a_focus_are_ignored_when_there_is_a_description(self, spawned):
        lessons.start_lesson({"topic": TOPIC, "class_code": "CHEM 1450", "start": "2026-09-09",
                              "end": "2026-09-09", "focus": "alcohols"})
        job = json.loads(spawned[0].read_text(encoding="utf-8"))
        assert job["class_code"] == lessons.FREESTYLE_CODE and job["focus"] == ""

    def test_a_notes_lesson_still_needs_its_class_and_dates(self, spawned):
        with pytest.raises(ValueError, match="start and end"):
            lessons.start_lesson({"class_code": "CHEM 1450"})
        lessons.start_lesson({"class_code": "CHEM 1450", "start": "2026-09-09", "end": "2026-09-09", "topic": None})
        job = json.loads(spawned[0].read_text(encoding="utf-8"))
        assert "topic" not in job and job["class_code"] == "CHEM 1450"

    def test_only_one_build_runs_at_a_time(self, library, monkeypatch):
        class Alive:
            def poll(self):
                return None
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: Alive())
        lessons.start_lesson({"topic": TOPIC})
        with pytest.raises(lessons.LessonBusy):
            lessons.start_lesson({"topic": "something else"})


class TestBuilding:
    def test_a_lesson_is_built_from_the_description_alone(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        lesson_id = build()
        record = lessons.get_lesson(lesson_id)
        assert lesson_id.startswith("Freestyle_")
        assert record["kind"] == "freestyle" and record["topic"] == TOPIC
        assert record["class_code"] == "Freestyle" and record["class_title"] == TOPIC
        assert record["dates"] == [TODAY.isoformat()] and record["focus"] == ""
        assert f'"{TOPIC}"' in fakes.prompts[0] and "NOTES (" not in fakes.prompts[0]
        assert len(record["slides"]) == len(PLAN["slides"])

    def test_it_works_with_no_notes_folder_at_all(self, library, fakes):
        for path in lessons.NOTES_DIR.glob("*"):
            path.unlink()
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        assert lessons.get_lesson(build())["title"] == PLAN["title"]

    def test_a_requested_length_reaches_the_prompt_and_the_record(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        record = lessons.get_lesson(build(minutes=1))
        assert "about 1 minutes" in " ".join(fakes.prompts[0].split()) and record["minutes"] == 1

    def test_a_notes_lesson_record_has_no_freestyle_fields(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        lesson_id = lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "af_heart")
        assert "kind" not in lessons.get_lesson(lesson_id) and "topic" not in lessons.get_lesson(lesson_id)

    def test_the_library_lists_kind_and_topic(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        build()
        listed = lessons.list_lessons()[0]
        assert listed["kind"] == "freestyle" and listed["topic"] == TOPIC


class TestRetry:
    @pytest.fixture
    def failed_once(self, library, fakes, monkeypatch):
        """A build whose narration step broke: the written plan stays saved for the next attempt."""
        available = lessons.speech.available

        def broken():
            raise RuntimeError("disk full")

        monkeypatch.setattr(lessons.speech, "available", broken)
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        with pytest.raises(RuntimeError):
            build()
        monkeypatch.setattr(lessons.speech, "available", available)
        return fakes

    @staticmethod
    def written(fakes):
        """The prompts that asked Claude to write a lesson (the picture picker has its own prompt)."""
        return [prompt for prompt in fakes.prompts if "outstanding tutor" in prompt]

    def test_the_same_request_reuses_the_saved_plan(self, failed_once):
        failed_once.replies = [json.dumps({"choices": {}})]
        lessons.get_lesson(build())
        assert len(self.written(failed_once)) == 1   # the retry did not write the lesson again

    def test_a_different_request_or_length_writes_a_new_plan(self, failed_once):
        failed_once.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        lessons.get_lesson(build(topic="how vaccines work"))
        written = self.written(failed_once)
        assert len(written) == 2 and "vaccines" in written[1] and TOPIC not in written[1]

    def test_the_page_can_tell_that_a_retry_would_reuse_the_plan(self, failed_once):
        job = {"class_code": "Freestyle", "start": TODAY.isoformat(), "end": TODAY.isoformat(), "topic": TOPIC}
        assert lessons._plan_saved(job) is True
        assert lessons._plan_saved(job | {"topic": "something else"}) is False
        assert lessons._plan_saved(job | {"minutes": 7}) is False

    def test_the_saved_plan_is_found_on_a_later_day(self, failed_once):
        tomorrow = date.fromordinal(TODAY.toordinal() + 1).isoformat()
        job = {"class_code": "Freestyle", "start": tomorrow, "end": tomorrow, "topic": TOPIC}
        assert lessons._plan_saved(job) is True


class TestProgress:
    def test_the_first_step_does_not_claim_to_read_notes(self, library):
        free = lessons._pipeline({"topic": TOPIC})
        notes = lessons._pipeline({})
        assert free[0][0] == "Getting ready" and notes[0][0] == "Reading notes"
        assert [step[0] for step in free[1:]] == [step[0] for step in notes[1:]]

    def test_a_builder_process_records_it(self, library, fakes, monkeypatch):
        spawned = []
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: spawned.append(path) or None)
        lessons.start_lesson({"topic": TOPIC})
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        assert lessons.run_job(spawned[0]) == 0
        job = json.loads(spawned[0].read_text(encoding="utf-8"))
        assert job["status"] == "ready" and job["steps"][0]["name"] == "Getting ready"
        assert lessons.get_lesson(job["lesson_id"])["topic"] == TOPIC


class TestRoute:
    @pytest.fixture
    def server(self, library, monkeypatch):
        monkeypatch.setattr(speech, "available", lambda: (True, ""))
        monkeypatch.setattr(dashboard.Handler, "log_message", lambda *a, **k: None)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield httpd.server_port
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    def post(self, port, payload):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/api/lessons/create", json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def test_the_server_says_it_can_build_from_a_description(self, server):
        """The page keeps Freestyle switched off on a dashboard that was started before this existed."""
        conn = http.client.HTTPConnection("127.0.0.1", server)
        conn.request("GET", "/api/lessons/classes")
        info = json.loads(conn.getresponse().read())
        conn.close()
        assert info["freestyle"] is True

    def test_a_description_starts_a_lesson_and_a_blank_one_is_a_400(self, server, monkeypatch):
        started = []
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: started.append(path) or None)
        status, data = self.post(server, {"topic": TOPIC, "voice": "af_heart", "minutes": ""})
        assert status == 200 and len(data["job_id"]) == 32 and len(started) == 1
        status, data = self.post(server, {"topic": "   "})
        assert status == 400 and "Describe what you want to learn" in data["error"]
