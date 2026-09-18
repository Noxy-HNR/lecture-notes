"""Lessons: note selection, plan validation, picture choice and the build pipeline.

Claude, the image sources and the narration voice are all faked - no network, model or
GPU - and everything is written under tmp_path.
"""
import http.client
import io
import json
import sys
import threading
import types
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from PIL import Image

import dashboard
import lessons
import speech

NOTES = """# College Chemistry I (CHEM 1450)

## Wednesday, September 2, 2026

### Atoms
- Protons are positive

## Friday, September 4, 2026

### Old material
- Only in early September

<!-- session:CHEM_1450_20260909_092036 -->
## Wednesday, September 9, 2026

### Functional groups
- Hydroxyl is OH
<!-- /session:CHEM_1450_20260909_092036 -->
## Wednesday, September 9, 2026

### More groups
- Carbonyl is C=O
"""

PLAN = {"title": "Functional groups", "tldr": "Groups decide how molecules behave.", "slides": [
    {"kind": "overview", "title": "What you'll learn", "bullets": ["Hydroxyl", "Carbonyl"],
     "narration": "Today we look at functional groups.", "visual": {"type": "none"}},
    {"kind": "concept", "title": "Hydroxyl", "bullets": ["An O-H group"], "narration": "An OH group makes alcohols.",
     "key_terms": [{"term": "Hydroxyl", "meaning": "an oxygen bonded to a hydrogen"}],
     "heads_up": "The notes said hydroxide; the group is hydroxyl.",
     "visual": {"type": "diagram", "mermaid": "flowchart LR\n  A[\"Alcohol\"] --> B[\"Hydroxyl\"]", "caption": "x"}},
    {"kind": "concept", "title": "Ethanol", "bullets": ["A simple alcohol"], "narration": "Ethanol is an alcohol.",
     "visual": {"type": "molecule", "compound": "ethanol", "caption": "ethanol"}},
    {"kind": "example", "title": "Spot it", "bullets": ["Look for O-H"], "narration": "Here is how to spot one.",
     "visual": {"type": "image", "queries": ["hydroxyl group"], "wikipedia": "Hydroxy group", "caption": "OH"}},
    {"kind": "check", "title": "Quick check", "narration": "Which group has a C double bond O?",
     "question": "Which group has C=O?", "answer": "Carbonyl, because it is a carbon double bonded to oxygen."},
]}

CANDIDATES = [
    {"url": "https://img.example/wrong.png", "source": "Wikimedia Commons", "title": "Unrelated",
     "description": "", "width": 800, "height": 600},
    {"url": "https://img.example/right.png", "source": "Wikipedia article image", "title": "Hydroxy group",
     "description": "", "width": 800, "height": 600},
]


@pytest.fixture
def library(tmp_path, monkeypatch):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "CHEM_1450.md").write_text(NOTES, encoding="utf-8")
    (notes_dir / "CHEM_1450_study_guide.md").write_text("# Guide\n\n## Topic\n", encoding="utf-8")
    monkeypatch.setattr(lessons, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(lessons, "LESSONS_DIR", tmp_path / "lessons")
    monkeypatch.setattr(lessons, "_processes", {})
    return tmp_path


def _png(size=(400, 300)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def fakes(monkeypatch):
    """Fake Claude (replies consumed in order), image search, downloads and voice."""
    state = types.SimpleNamespace(replies=[], prompts=[], fetched=[])

    def ask(prompt, timeout):
        state.prompts.append(prompt)
        return state.replies.pop(0)

    def get(url, accept="*/*", limit=0):
        state.fetched.append(url)
        return _png(), "image/png"

    def tts(text, path, voice=speech.DEFAULT_VOICE, speed=1.0):
        path.write_bytes(b"OggS")
        return 10.0

    monkeypatch.setattr(lessons, "_ask_claude", ask)
    monkeypatch.setattr(lessons, "image_candidates", lambda visual: list(CANDIDATES))
    monkeypatch.setattr(lessons, "_http_get", get)
    monkeypatch.setattr(lessons.speech, "available", lambda: (True, ""))
    monkeypatch.setattr(lessons.speech, "synthesize", tts)
    return state


class TestNoteSelection:
    def test_range_keeps_each_date_heading_once_without_session_markers(self, library):
        title, body, dates = lessons.notes_for_range("CHEM 1450", date(2026, 9, 5), date(2026, 9, 9))
        assert title == "College Chemistry I" and dates == ["2026-09-09"]
        assert body.count("## Wednesday, September 9, 2026") == 1
        assert "Hydroxyl is OH" in body and "Carbonyl is C=O" in body
        assert "early September" not in body and "session:" not in body

    def test_range_is_inclusive_and_empty_or_invalid_requests_fail(self, library):
        assert lessons.notes_for_range("CHEM 1450", date(2026, 9, 2), date(2026, 9, 4))[2] == ["2026-09-02", "2026-09-04"]
        with pytest.raises(ValueError):
            lessons.notes_for_range("CHEM 1450", date(2026, 9, 10), date(2026, 9, 12))
        with pytest.raises(ValueError):
            lessons.notes_for_range("../../secrets", date(2026, 9, 2), date(2026, 9, 4))

    def test_classes_list_lecture_dates_and_skip_study_guides(self, library):
        assert lessons.list_classes() == [{"code": "CHEM 1450", "title": "College Chemistry I",
                                           "dates": ["2026-09-02", "2026-09-04", "2026-09-09"]}]


class TestLessonPlan:
    def test_prompt_scales_with_notes_and_treats_notes_as_data(self):
        small = lessons.build_lesson_prompt("Chemistry", "CHEM 1450", "2026-09-09", "2026-09-09", "word " * 100)
        large = lessons.build_lesson_prompt("Chemistry", "CHEM 1450", "2026-09-02", "2026-09-16", "word " * 9000)
        assert "6-8 slides" in small and "6-24 slides" in large
        assert "heads_up" in small and "not instructions" in small and "Go well beyond the notes" in small

    def test_malformed_slides_and_unsafe_diagrams_are_dropped(self):
        raw = dict(PLAN, slides=PLAN["slides"] + [
            {"title": "", "narration": "a slide without a title"},
            {"kind": "check", "title": "No answer", "narration": "Question?", "question": "Question?"},
            {"kind": "weird", "title": "Unsafe diagram", "narration": "Careful.",
             "visual": {"type": "diagram", "mermaid": "flowchart LR\n  A-->B\n  click A \"javascript:alert(1)\""}},
            {"title": "Bad molecule", "narration": "n", "visual": {"type": "molecule", "compound": "<img src=x>"}},
            "not a slide",
        ])
        lesson = lessons.parse_lesson("Here you go:\n```json\n" + json.dumps(raw) + "\n```")
        assert [s["title"] for s in lesson["slides"]] == [
            "What you'll learn", "Hydroxyl", "Ethanol", "Spot it", "Quick check", "Unsafe diagram", "Bad molecule"]
        assert lesson["slides"][1]["visual"]["type"] == "diagram"
        assert lesson["slides"][5]["kind"] == "concept" and lesson["slides"][5]["visual"] == {"type": "none"}
        assert lesson["slides"][6]["visual"] == {"type": "none"}

    def test_a_plan_without_usable_slides_is_rejected(self):
        with pytest.raises(ValueError):
            lessons.parse_lesson('{"title": "Empty", "slides": []}')
        with pytest.raises(ValueError):
            lessons.parse_lesson("I couldn't do that.")


class TestBuild:
    def test_build_writes_a_complete_lesson(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {"4": 2}})]
        lesson_id = lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "am_michael")
        lesson = lessons.get_lesson(lesson_id)

        assert "Carbonyl is C=O" in fakes.prompts[0]              # the notes reached Claude
        assert "Hydroxy group" in fakes.prompts[1]                 # candidates reached the picker
        assert lesson["dates"] == ["2026-09-09"] and lesson["voice"] == "am_michael"
        assert len(lesson["slides"]) == 5 and lesson["duration_seconds"] == 50.0 and lesson["warnings"] == []
        molecule, picture = lesson["slides"][2]["visual"], lesson["slides"][3]["visual"]
        assert molecule["type"] == "molecule" and molecule["file"] == "slide-03.webp"
        assert picture == {"type": "image", "caption": "OH", "file": "slide-04.webp",
                           "source_url": "https://img.example/right.png"}
        assert "https://img.example/wrong.png" not in fakes.fetched
        assert all(lessons.lesson_asset(lesson_id, s["audio"]).is_file() for s in lesson["slides"])
        assert [entry["id"] for entry in lessons.list_lessons()] == [lesson_id]
        assert not list(lessons.LESSONS_DIR.glob("*.partial"))

    def test_picker_saying_none_fits_leaves_the_slide_without_a_picture(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {"4": None}})]
        lesson = lessons.get_lesson(lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "af_heart"))
        assert lesson["slides"][3]["visual"] == {"type": "none"}

    def test_unusable_picker_only_trusts_the_wikipedia_article_image(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), "sorry, no JSON"]
        lesson = lessons.get_lesson(lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "af_heart"))
        assert lesson["slides"][3]["visual"]["source_url"] == "https://img.example/right.png"
        assert any("Picture choice failed" in w for w in lesson["warnings"])

    def test_invalid_plan_is_retried_once(self, library, fakes):
        fakes.replies = ["not json at all", json.dumps(PLAN), json.dumps({"choices": {}})]
        lesson_id = lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "af_heart")
        assert "not valid JSON" in fakes.prompts[1] and lessons.get_lesson(lesson_id)["title"] == "Functional groups"

    def test_lesson_without_a_voice_is_still_built(self, library, fakes, monkeypatch):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        monkeypatch.setattr(lessons.speech, "available", lambda: (False, "voice model files not found"))
        lesson = lessons.get_lesson(lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "af_heart"))
        assert all("audio" not in s for s in lesson["slides"]) and lesson["duration_seconds"] == 0
        assert lesson["warnings"] == ["No narration: voice model files not found"]

    def test_failed_build_leaves_no_lesson_but_keeps_the_plan_for_a_retry(self, library, fakes, monkeypatch):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        available = lessons.speech.available

        def broken():
            raise RuntimeError("disk full")

        monkeypatch.setattr(lessons.speech, "available", broken)
        with pytest.raises(RuntimeError):
            lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "af_heart")
        assert lessons.list_lessons() == [] and not list(lessons.LESSONS_DIR.glob("*.partial"))
        assert len(list((lessons.LESSONS_DIR / ".plans").glob("*.json"))) == 1

        monkeypatch.setattr(lessons.speech, "available", available)
        fakes.replies = [json.dumps({"choices": {}})]  # only the picture picker: no second lesson plan
        lesson_id = lessons.build_lesson("CHEM 1450", date(2026, 9, 9), date(2026, 9, 9), "af_heart")
        # First build: plan + picture picker. Retry: the picker only - the plan came from the cache.
        assert len(fakes.prompts) == 3 and "outstanding tutor" not in fakes.prompts[2]
        assert lessons.get_lesson(lesson_id)["title"] == "Functional groups"
        assert not list((lessons.LESSONS_DIR / ".plans").glob("*.json"))  # cleared once built

    def test_long_narration_is_written_in_blocks(self, tmp_path, monkeypatch):
        """libsndfile's Vorbis encoder crashed natively when given ~25s+ of audio in one write."""
        writes = []

        class FakeKokoro:
            def __init__(self, model, voices):
                pass

            def create(self, text, voice, speed, lang):
                return np.zeros(24000 * 40, dtype=np.float32), 24000  # 40 seconds

        import soundfile
        original = soundfile.SoundFile.write

        def spy(self, data):
            writes.append(len(data))
            return original(self, data)

        monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
        monkeypatch.setattr(speech, "_engine", None)
        monkeypatch.setattr(soundfile.SoundFile, "write", spy)
        assert speech.synthesize("One long sentence.", tmp_path / "slide-01.ogg") == 40.0
        assert max(writes) <= speech.WRITE_BLOCK_FRAMES and sum(writes) == 24000 * 40

    def test_builds_run_in_their_own_process_one_at_a_time(self, library, monkeypatch):
        class FakeProcess:
            returncode = None

            def poll(self):
                return self.returncode

        spawned = []
        monkeypatch.setattr(lessons, "_spawn_builder", lambda path: spawned.append(path) or FakeProcess())
        leftover = lessons.LESSONS_DIR / "CHEM_1450_20260909-20260909_20260909120000.partial"
        leftover.mkdir(parents=True)
        payload = {"class_code": "CHEM 1450", "start": "2026-09-09", "end": "2026-09-02"}

        job_id = lessons.start_lesson(payload)["job_id"]
        job = lessons.get_job(job_id)
        assert len(spawned) == 1 and not leftover.exists()
        assert job["status"] == "running" and (job["start"], job["end"]) == ("2026-09-02", "2026-09-09")
        with pytest.raises(lessons.LessonBusy):
            lessons.start_lesson(payload)

        lessons._processes[job_id].returncode = 3221225477  # the builder dies without reporting
        job = lessons.get_job(job_id)
        assert job["status"] == "failed" and "stopped unexpectedly (exit code 3221225477)" in job["error"]
        assert lessons.start_lesson(payload)["job_id"] != job_id  # no longer busy

    def test_builder_process_reports_progress_and_the_finished_lesson(self, library, fakes):
        fakes.replies = [json.dumps(PLAN), json.dumps({"choices": {}})]
        status = library / "job.json"
        status.write_text(json.dumps({"id": "a" * 32, "status": "running", "class_code": "CHEM 1450",
                                      "start": "2026-09-09", "end": "2026-09-09", "voice": "bf_emma",
                                      "created_at": 0}), encoding="utf-8")
        assert lessons.run_job(status) == 0
        job = json.loads(status.read_text(encoding="utf-8"))
        assert job["status"] == "ready" and job["progress"] == 1.0 and "status_path" not in job
        assert lessons.get_lesson(job["lesson_id"])["voice"] == "bf_emma"

    def test_builder_process_reports_a_failed_build(self, library, fakes):
        fakes.replies = ["not a plan", "still not a plan"]
        status = library / "job.json"
        status.write_text(json.dumps({"id": "b" * 32, "status": "running", "class_code": "CHEM 1450",
                                      "start": "2026-09-09", "end": "2026-09-09", "created_at": 0}), encoding="utf-8")
        assert lessons.run_job(status) == 1
        job = json.loads(status.read_text(encoding="utf-8"))
        assert job["status"] == "failed" and job["error"]


class TestPictures:
    def test_transparent_figures_are_flattened_onto_white(self, tmp_path):
        """Textbook figures are often black lines on transparency - invisible on a dark slide."""
        figure = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        figure.paste((0, 0, 0, 255), (100, 90, 200, 110))  # a black bar
        buffer = io.BytesIO()
        figure.save(buffer, "PNG")
        stored = lessons._store_image(buffer.getvalue(), tmp_path / "slide-01")
        with Image.open(stored) as result:
            assert result.mode == "RGB" and result.getpixel((5, 5)) == (255, 255, 255)
            assert sum(result.getpixel((150, 100))) < 60

    def test_raster_molecule_fallback_is_cropped_and_enlarged(self, tmp_path):
        """PubChem draws a small molecule as a speck on a light-gray (not white) square."""
        sketch = Image.new("RGB", (600, 600), (245, 245, 245))
        sketch.paste((20, 20, 20), (280, 285, 320, 315))
        buffer = io.BytesIO()
        sketch.save(buffer, "PNG")
        with Image.open(lessons._store_image(buffer.getvalue(), tmp_path / "slide-02", trim=True)) as result:
            assert max(result.size) == 480 and result.width > result.height

    def test_molecules_are_drawn_as_vectors_from_pubchem_structures(self, tmp_path, monkeypatch):
        pytest.importorskip("rdkit")
        asked = []
        monkeypatch.setattr(lessons, "_get_json", lambda url: asked.append(url) or
                            {"PropertyTable": {"Properties": [{"CID": 962, "SMILES": "O"}]}})
        drawn = lessons._molecule_image("water", tmp_path / "slide-04")
        assert drawn.name == "slide-04.svg" and "/compound/name/water/property/SMILES/JSON" in asked[0]
        svg = drawn.read_text(encoding="utf-8")
        assert svg.lstrip().startswith("<?xml") and "<script" not in svg and "atom-2" in svg  # H, O, H

    def test_unknown_structure_falls_back_to_pubchem_depiction(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lessons, "_get_json", lambda url: {"Fault": {"Code": "PUGREST.NotFound"}})
        monkeypatch.setattr(lessons, "_pubchem_depiction", lambda name, path: path.with_suffix(".webp"))
        assert lessons._molecule_image("mystery", tmp_path / "slide-05").suffix == ".webp"

    def test_tiny_images_are_rejected(self, tmp_path):
        buffer = io.BytesIO()
        Image.new("RGB", (60, 40), (10, 10, 10)).save(buffer, "PNG")
        with pytest.raises(ValueError):
            lessons._store_image(buffer.getvalue(), tmp_path / "slide-03")


class TestDashboardRoutes:
    @pytest.fixture
    def server(self, library, monkeypatch):
        monkeypatch.setattr(speech, "available", lambda: (True, ""))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield httpd.server_port
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    def _request(self, port, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request(method, path, body, headers or {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def test_classes_and_create_with_request_checks(self, server, monkeypatch):
        started = []
        monkeypatch.setattr(lessons, "start_lesson", lambda payload: started.append(payload) or {"job_id": "abc"})
        status, data = self._request(server, "GET", "/api/lessons/classes")
        assert status == 200 and json.loads(data)["classes"][0]["code"] == "CHEM 1450"

        body = json.dumps({"class_code": "CHEM 1450", "start": "2026-09-09", "end": "2026-09-09"})
        assert self._request(server, "POST", "/api/lessons/create", body, {"Content-Type": "text/plain"})[0] == 415
        rebinding = f"attacker.example:{server}"
        assert self._request(server, "POST", "/api/lessons/create", body,
                             {"Content-Type": "application/json", "Host": rebinding,
                              "Origin": "http://" + rebinding})[0] == 403
        status, data = self._request(server, "POST", "/api/lessons/create", body, {"Content-Type": "application/json"})
        assert status == 200 and json.loads(data) == {"job_id": "abc"} and len(started) == 1

    def test_lesson_files_cannot_escape_the_lessons_folder(self, server):
        for path in ("/lesson-assets?id=..%2F..%2Fstate&file=status.json",
                     "/lesson-assets?id=CHEM_1450_x&file=..%2Flesson.json",
                     "/api/lessons/lesson?id=..%2Fnotes"):
            assert self._request(server, "GET", path)[0] == 404


class TestSpeech:
    def test_speakable_removes_symbols_a_voice_would_read_aloud(self):
        assert speech.speakable("**Glucose** → pyruvate & 2 ATP (95% of cells)") == \
            "Glucose to pyruvate and 2 ATP (95 percent of cells)"

    def test_long_narration_is_spoken_in_pieces_kokoro_can_handle(self, tmp_path, monkeypatch):
        """Kokoro killed the whole process on ~500 characters of text in one call."""
        calls = []

        class FakeKokoro:
            def __init__(self, model, voices):
                pass

            def create(self, text, voice, speed, lang):
                assert len(text) <= speech.MAX_CHUNK_CHARS
                calls.append(text)
                return np.ones(2400, dtype=np.float32), 24000

        monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
        monkeypatch.setattr(speech, "_engine", None)
        text = ("Water molecules stick together because oxygen pulls electrons closer than hydrogen does, "
                "which leaves the oxygen end slightly negative and the hydrogen ends slightly positive. ") * 6
        seconds = speech.synthesize(text, tmp_path / "slide-01.ogg")
        assert len(calls) >= 4 and " ".join(calls).split() == speech.speakable(text).split()
        assert seconds == pytest.approx(len(calls) * 0.1 + (len(calls) - 1) * speech.CHUNK_PAUSE_SECONDS)

    def test_chunks_split_at_sentences_then_commas_then_anywhere(self):
        run_on = ", ".join(["a clause about the carbon cycle"] * 30)
        assert all(len(p) <= 250 for p in speech.chunks(run_on))
        assert " ".join(speech.chunks(run_on)).split() == run_on.split()
        assert speech.chunks("One. Two. Three.") == ["One. Two. Three."]
        unbroken = "x" * 600
        assert all(len(p) <= 250 for p in speech.chunks(unbroken)) and "".join(speech.chunks(unbroken)) == unbroken

    def test_synthesize_writes_ogg_and_returns_seconds(self, tmp_path, monkeypatch):
        class FakeKokoro:
            def __init__(self, model, voices):
                pass

            def create(self, text, voice, speed, lang):
                assert voice == "bf_emma" and lang == "en-us" and "*" not in text
                return np.zeros(24000 * 2, dtype=np.float32), 24000

        monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
        monkeypatch.setattr(speech, "_engine", None)
        out = tmp_path / "slide-01.ogg"
        assert speech.synthesize("**Hello** there", out, voice="bf_emma") == 2.0
        assert out.read_bytes()[:4] == b"OggS"
