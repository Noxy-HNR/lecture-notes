"""Quiz mode: the saved results, the review queue and how bad input is refused.

Everything is written under tmp_path; no model is involved.
"""
import http.client
import json
import sys
import threading
import types
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import dashboard
import lessons
import quiz

CHECKS = [
    {"kind": "overview", "title": "What you'll learn", "bullets": ["Groups"], "narration": "Hi."},
    {"kind": "check", "title": "Check 1", "narration": "Which group has C=O?",
     "question": "Which group has C=O?", "answer": "Carbonyl."},
    {"kind": "concept", "title": "Hydroxyl", "bullets": ["O-H"], "narration": "An OH group."},
    {"kind": "check", "title": "Check 2", "narration": "What does OH make?",
     "question": "What does a hydroxyl make?", "answer": "Alcohols."},
    {"kind": "recap", "title": "Recap", "bullets": ["Done"], "narration": "Done."},
]
LESSON_ID = "CHEM_1450_20260909-20260909_20260909100000"
OTHER_ID = "BIOL_1440_20260901-20260922_20260924110020"


def make_lesson(root: Path, lesson_id=LESSON_ID, title="Functional groups", slides=None, code="CHEM 1450"):
    folder = root / lesson_id
    folder.mkdir(parents=True)
    (folder / "lesson.json").write_text(json.dumps({
        "id": lesson_id, "class_code": code, "title": title,
        "slides": CHECKS if slides is None else slides}), encoding="utf-8")


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "lessons"
    root.mkdir()
    monkeypatch.setattr(lessons, "LESSONS_DIR", root)
    make_lesson(root)
    return root


def clock(monkeypatch, start=1000.0):
    """time.time() that moves one second per call, so 'most recently missed' is exact."""
    now = [start]

    def tick():
        now[0] += 1
        return now[0]
    monkeypatch.setattr(quiz, "time", types.SimpleNamespace(time=tick))


def answer(slide, result, typed=None, lesson_id=LESSON_ID):
    payload = {"lesson_id": lesson_id, "slide": slide, "result": result}
    if typed is not None:
        payload["typed"] = typed
    return quiz.record_answer(payload)


def by_slide(state):
    return {q["slide"]: q for q in state["questions"]}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_fresh_state_lists_only_the_check_slides(library):
    state = quiz.get_state(LESSON_ID)
    assert state["lesson_id"] == LESSON_ID
    assert state["awaiting_review"] == 0
    assert [q["slide"] for q in state["questions"]] == [2, 4]      # 1-based positions in the lesson
    assert by_slide(state)[2] == {"slide": 2, "question": "Which group has C=O?", "answer": "Carbonyl.",
                                  "attempts": 0, "got": 0, "missed": 0, "awaiting_review": False,
                                  "last_result": None, "last_typed": ""}


def test_counts_and_review_flag_follow_the_results(library):
    answer(2, "missed")
    answer(2, "missed")
    state = answer(2, "got")
    first = by_slide(state)[2]
    assert (first["attempts"], first["got"], first["missed"]) == (3, 1, 2)
    assert first["last_result"] == "got"
    assert first["awaiting_review"] is False        # answering it right clears it from review
    assert state["awaiting_review"] == 0
    state = answer(4, "missed")
    assert by_slide(state)[4]["awaiting_review"] is True
    assert state["awaiting_review"] == 1
    assert by_slide(state)[2]["attempts"] == 3      # the other question is untouched


def test_a_miss_after_a_correct_answer_puts_it_back_in_review(library):
    answer(2, "got")
    state = answer(2, "missed")
    assert by_slide(state)[2]["awaiting_review"] is True
    assert by_slide(state)[2]["last_result"] == "missed"


def test_results_survive_a_restart(library):
    answer(2, "missed", typed="oxygen?")
    again = quiz.get_state(LESSON_ID)               # state is read back from the file every time
    assert by_slide(again)[2]["missed"] == 1
    assert by_slide(again)[2]["last_typed"] == "oxygen?"
    assert json.loads((library / ".quiz" / f"{LESSON_ID}.json").read_text(encoding="utf-8"))["lesson_id"] == LESSON_ID


def test_typed_text_is_stored_verbatim_as_data(library):
    state = answer(2, "missed", typed="<b>x</b> & <script>alert(1)</script>")
    assert by_slide(state)[2]["last_typed"] == "<b>x</b> & <script>alert(1)</script>"   # the page escapes it on show


def test_a_lesson_without_check_slides_has_an_empty_quiz(library):
    make_lesson(library, OTHER_ID, slides=[CHECKS[0], CHECKS[2]])
    assert quiz.get_state(OTHER_ID)["questions"] == []


def test_a_reworded_question_does_not_inherit_the_old_result(library):
    answer(2, "missed")
    path = library / LESSON_ID / "lesson.json"
    lesson = json.loads(path.read_text(encoding="utf-8"))
    lesson["slides"][1]["question"] = "A different question entirely?"
    path.write_text(json.dumps(lesson), encoding="utf-8")
    state = quiz.get_state(LESSON_ID)
    assert by_slide(state)[2]["attempts"] == 0 and by_slide(state)[2]["awaiting_review"] is False
    assert quiz.review() == {"count": 0, "items": []}


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

def test_review_is_empty_without_any_results(library):
    assert quiz.review() == {"count": 0, "items": []}


def test_review_lists_missed_questions_most_recently_missed_first(library, monkeypatch):
    make_lesson(library, OTHER_ID, title="Cell membranes", code="BIOL 1440")
    clock(monkeypatch)
    answer(2, "missed")                                       # t=1001
    answer(4, "missed", lesson_id=OTHER_ID)                   # t=1002
    answer(4, "missed")                                       # t=1003
    answer(2, "missed", lesson_id=OTHER_ID)                   # t=1004
    answer(4, "got", lesson_id=OTHER_ID)                      # answered right: leaves the queue
    result = quiz.review()
    assert result["count"] == 3
    assert [(i["lesson_id"], i["slide"]) for i in result["items"]] == [
        (OTHER_ID, 2), (LESSON_ID, 4), (LESSON_ID, 2)]
    newest = result["items"][0]
    assert newest == {"lesson_id": OTHER_ID, "lesson_title": "Cell membranes", "class_code": "BIOL 1440",
                      "slide": 2, "question": "Which group has C=O?", "answer": "Carbonyl.",
                      "missed": 1, "last_missed_at": 1004.0}


def test_review_counts_how_often_a_question_was_missed(library):
    answer(2, "missed")
    answer(2, "missed")
    assert quiz.review()["items"][0]["missed"] == 2


def test_review_skips_a_lesson_that_was_deleted(library):
    make_lesson(library, OTHER_ID, slides=[CHECKS[0], CHECKS[1]])
    answer(2, "missed")
    answer(2, "missed", lesson_id=OTHER_ID)
    import shutil
    shutil.rmtree(library / OTHER_ID)
    result = quiz.review()
    assert [i["lesson_id"] for i in result["items"]] == [LESSON_ID]
    assert result["count"] == 1
    with pytest.raises(FileNotFoundError):
        quiz.get_state(OTHER_ID)


def test_review_ignores_stray_files_in_the_quiz_folder(library):
    answer(2, "missed")
    quiz_dir = library / ".quiz"
    (quiz_dir / "notes.txt").write_text("hello", encoding="utf-8")
    (quiz_dir / "bad name!.json").write_text("{}", encoding="utf-8")
    (quiz_dir / f"{OTHER_ID}.json.tmp").write_text("{", encoding="utf-8")
    assert quiz.review()["count"] == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lesson_id", ["../x", "..", "a/b", "a\\b", "", ".quiz", ".jobs", None, 7, ["x"], "nope",
                                       LESSON_ID + "/../" + LESSON_ID, "CHEM 1450"])
def test_bad_or_unknown_lesson_ids_are_refused(library, lesson_id):
    with pytest.raises(FileNotFoundError):
        quiz.get_state(lesson_id)
    with pytest.raises(FileNotFoundError):
        quiz.record_answer({"lesson_id": lesson_id, "slide": 2, "result": "got"})
    assert not (library / ".quiz").exists()                   # and nothing was written


def test_a_traversing_reset_is_refused_and_removes_nothing(library, tmp_path):
    answer(2, "missed")
    outside = tmp_path / "keep.json"
    outside.write_text("{}", encoding="utf-8")
    for bad in ("../keep", "..\\keep", "../../keep", str(outside)):
        with pytest.raises(FileNotFoundError):
            quiz.reset({"lesson_id": bad})
    assert outside.exists() and (library / ".quiz" / f"{LESSON_ID}.json").exists()


@pytest.mark.parametrize("slide", [0, -1, 1, 3, 5, 99, True, False, "2", 2.0, None])
def test_only_real_check_slides_can_be_answered(library, slide):
    # 1 is the overview, 3 a concept, 5 the recap; 0 would index the last slide, and True is 1 in Python
    with pytest.raises(ValueError):
        quiz.record_answer({"lesson_id": LESSON_ID, "slide": slide, "result": "got"})
    assert not (library / ".quiz").exists()


@pytest.mark.parametrize("result", ["GOT", "right", "", None, True, 1, ["got"]])
def test_the_result_must_be_got_or_missed(library, result):
    with pytest.raises(ValueError):
        quiz.record_answer({"lesson_id": LESSON_ID, "slide": 2, "result": result})


def test_typed_text_has_a_limit_and_must_be_text(library):
    assert by_slide(answer(2, "got", typed="x" * 2000))[2]["last_typed"] == "x" * 2000
    with pytest.raises(ValueError):
        answer(2, "got", typed="x" * 2001)
    with pytest.raises(ValueError):
        quiz.record_answer({"lesson_id": LESSON_ID, "slide": 2, "result": "got", "typed": 5})
    assert quiz.get_state(LESSON_ID)["questions"][0]["attempts"] == 1     # only the valid one counted


def test_a_check_slide_missing_its_question_or_answer_is_not_a_quiz_question(library):
    broken = [{"kind": "check", "title": "No answer", "question": "Why?", "narration": "x"},
              {"kind": "check", "title": "Blank", "question": "  ", "answer": "y", "narration": "x"},
              {"kind": "check", "title": "Numbers", "question": 4, "answer": 5, "narration": "x"},
              CHECKS[3]]
    make_lesson(library, OTHER_ID, slides=broken)
    assert [q["slide"] for q in quiz.get_state(OTHER_ID)["questions"]] == [4]
    with pytest.raises(ValueError):
        answer(1, "got", lesson_id=OTHER_ID)


# ---------------------------------------------------------------------------
# Damaged files and persistence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    "", "not json at all", "{", "[]", "null", "7", '{"answers": []}', '{"answers": "x"}',
    '{"answers": {"2": 5, "4": []}}', '{"answers": {"x": {"attempts": "lots", "missed": -3, "awaiting_review": "yes"}}}',
    '{"answers": {"2": {"question": "Which group has C=O?", "attempts": 1e9, "got": true, "last_result": "maybe"}}}',
    b"\xff\xfe\x00bad bytes".decode("latin-1")])
def test_a_corrupt_quiz_file_counts_as_no_history(library, content):
    (library / ".quiz").mkdir()
    (library / ".quiz" / f"{LESSON_ID}.json").write_text(content, encoding="utf-8")
    state = quiz.get_state(LESSON_ID)
    assert state["awaiting_review"] == 0
    assert all(q["attempts"] == 0 and q["got"] == 0 and q["missed"] == 0 for q in state["questions"])
    assert quiz.review() == {"count": 0, "items": []}
    assert by_slide(answer(2, "missed"))[2]["missed"] == 1             # and the next answer repairs the file


def test_a_quiz_file_that_is_not_valid_utf8_does_not_crash(library):
    (library / ".quiz").mkdir()
    (library / ".quiz" / f"{LESSON_ID}.json").write_bytes(b"\xff\xfe{")
    assert quiz.get_state(LESSON_ID)["awaiting_review"] == 0
    assert quiz.review()["count"] == 0


def test_writes_are_atomic_and_leave_no_temp_file(library):
    answer(2, "missed")
    names = sorted(p.name for p in (library / ".quiz").iterdir())
    assert names == [f"{LESSON_ID}.json"]


def test_a_failed_write_leaves_the_previous_results_intact(library, monkeypatch):
    answer(2, "missed")
    path = library / ".quiz" / f"{LESSON_ID}.json"
    before = path.read_text(encoding="utf-8")

    def broken(src, dst):
        raise OSError("disk full")
    with monkeypatch.context() as patch:
        patch.setattr(quiz, "os", types.SimpleNamespace(replace=broken))
        with pytest.raises(OSError):
            answer(2, "got")
    assert path.read_text(encoding="utf-8") == before            # replace is the only step that touches the real file
    assert by_slide(quiz.get_state(LESSON_ID))[2]["awaiting_review"] is True


def test_a_failed_write_leaves_no_temp_file_behind(library, monkeypatch):
    answer(2, "missed")

    def broken(src, dst):
        raise OSError("file in use")
    with monkeypatch.context() as patch:
        patch.setattr(quiz, "os", types.SimpleNamespace(replace=broken))
        with pytest.raises(OSError):
            answer(2, "got")
    assert sorted(p.name for p in (library / ".quiz").iterdir()) == [f"{LESSON_ID}.json"]


def test_a_lone_surrogate_in_the_typed_text_is_refused_plainly(library):
    with pytest.raises(ValueError, match="can't be saved"):
        answer(2, "missed", typed="\ud800")
    assert not (library / ".quiz").exists()          # nothing was written, not even a temp file
    assert by_slide(quiz.get_state(LESSON_ID))[2]["attempts"] == 0


def test_a_read_waits_for_a_write_in_progress(library):
    answer(2, "missed")
    finished = threading.Event()
    with quiz._lock:                                  # a request that is in the middle of saving
        reader = threading.Thread(target=lambda: (quiz.get_state(LESSON_ID), finished.set()))
        reader.start()
        assert not finished.wait(0.3)                 # the read is held back until the write is done
    assert finished.wait(5)
    reader.join()


def test_simultaneous_answers_are_all_counted(library):
    errors = []

    def worker():
        try:
            for _ in range(10):
                answer(2, "missed")
        except Exception as error:      # noqa: BLE001 - reported below
            errors.append(error)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert by_slide(quiz.get_state(LESSON_ID))[2]["attempts"] == 80
    json.loads((library / ".quiz" / f"{LESSON_ID}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_one_lesson_only(library):
    make_lesson(library, OTHER_ID, slides=[CHECKS[0], CHECKS[1]])
    answer(2, "missed")
    answer(2, "missed", lesson_id=OTHER_ID)
    assert quiz.reset({"lesson_id": LESSON_ID}) == {"cleared": 1}
    assert quiz.get_state(LESSON_ID)["awaiting_review"] == 0
    assert by_slide(quiz.get_state(OTHER_ID))[2]["awaiting_review"] is True
    assert quiz.reset({"lesson_id": LESSON_ID}) == {"cleared": 0}      # nothing left is not an error


def test_reset_all_needs_an_explicit_all_true(library):
    answer(2, "missed")
    for payload in ({}, {"all": False}, {"all": "yes"}, {"all": 1}, {"unrelated": True}):
        with pytest.raises(ValueError):
            quiz.reset(payload)
    assert quiz.review()["count"] == 1
    assert quiz.reset({"all": True}) == {"cleared": 1}
    assert quiz.review()["count"] == 0
    assert library.is_dir() and (library / LESSON_ID / "lesson.json").exists()   # only quiz files go


def test_reset_all_without_a_quiz_folder_is_fine(library):
    assert quiz.reset({"all": True}) == {"cleared": 0}


def test_reset_can_clear_the_leftovers_of_a_deleted_lesson(library):
    answer(2, "missed")
    import shutil
    shutil.rmtree(library / LESSON_ID)
    assert quiz.reset({"lesson_id": LESSON_ID}) == {"cleared": 1}
    assert not (library / ".quiz" / f"{LESSON_ID}.json").exists()


# ---------------------------------------------------------------------------
# Through the real dashboard routes
# ---------------------------------------------------------------------------

@pytest.fixture
def served(library, monkeypatch):
    monkeypatch.setattr(dashboard.Handler, "log_message", lambda *a, **k: None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_port
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def call(port, method, path, payload=None):
    headers = {"X-Notes-Dashboard": "1", "Content-Type": "application/json"} if method == "POST" else {}
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request(method, path, json.dumps(payload) if payload is not None else None, headers)
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    return response.status, body


def test_the_quiz_works_end_to_end_over_http(served):
    status, state = call(served, "GET", f"/api/quiz?lesson={LESSON_ID}")
    assert status == 200 and len(state["questions"]) == 2
    status, state = call(served, "POST", "/api/quiz/answer",
                         {"lesson_id": LESSON_ID, "slide": 4, "result": "missed", "typed": "<i>hm</i>"})
    assert status == 200 and state["awaiting_review"] == 1
    status, queue = call(served, "GET", "/api/quiz/review")
    assert status == 200 and queue["count"] == 1 and queue["items"][0]["slide"] == 4
    status, _ = call(served, "POST", "/api/quiz/answer", {"lesson_id": LESSON_ID, "slide": 4, "result": "got"})
    assert status == 200 and call(served, "GET", "/api/quiz/review")[1]["count"] == 0
    assert call(served, "POST", "/api/quiz/reset", {"lesson_id": LESSON_ID}) == (200, {"cleared": 1})


def test_bad_quiz_requests_get_plain_errors_over_http(served):
    status, body = call(served, "POST", "/api/quiz/answer", {"lesson_id": LESSON_ID, "slide": 1, "result": "got"})
    assert status == 400 and "quiz question" in body["error"]
    status, body = call(served, "GET", "/api/quiz?lesson=../../etc")
    assert status == 404 and body["error"] == "Lesson not found"
    status, body = call(served, "GET", "/api/quiz")
    assert status == 404
    status, body = call(served, "POST", "/api/quiz/reset", {})
    assert status == 400
