"""Quiz mode for lessons: the student answers a lesson's check slides and grades themselves.

Nothing here calls a model and no questions are generated: the questions and answers are the
ones already in lesson.json. What a student did is kept per lesson in lessons/.quiz/<id>.json,
so a missed question can come back later in the review queue until it is answered correctly.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

import lessons

MAX_TYPED = 2000
RESULTS = ("got", "missed")

# The dashboard serves requests on threads, and an answer is a read-modify-write of one file. Reads take
# the lock too: on Windows a replace fails while another thread has the file open for reading.
# Reentrant, because record_answer holds it while it loads the file.
_lock = threading.RLock()


def _quiz_dir() -> Path:
    return lessons.LESSONS_DIR / ".quiz"      # looked up per call: tests point LESSONS_DIR elsewhere


def _check_id(lesson_id) -> str:
    """The same shape lessons._lesson_dir accepts, so the id is safe to use in a file name."""
    if not isinstance(lesson_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", lesson_id):
        raise FileNotFoundError("Lesson not found")
    return lesson_id


def _quiz_path(lesson_id: str) -> Path:
    return _quiz_dir() / f"{_check_id(lesson_id)}.json"


def _count(value) -> int:
    return value if type(value) is int and value > 0 else 0


def _clean_record(raw) -> dict:
    """One question's saved history, with anything unexpected dropped: a hand-edited or
    half-damaged file must not take the lesson's quiz down."""
    raw = raw if isinstance(raw, dict) else {}
    missed_at = raw.get("last_missed_at")
    return {
        "question": raw.get("question") if isinstance(raw.get("question"), str) else "",
        "attempts": _count(raw.get("attempts")), "got": _count(raw.get("got")), "missed": _count(raw.get("missed")),
        "last_result": raw.get("last_result") if raw.get("last_result") in RESULTS else None,
        "awaiting_review": raw.get("awaiting_review") is True,
        "last_missed_at": float(missed_at) if isinstance(missed_at, (int, float)) and not isinstance(missed_at, bool) else 0.0,
        "last_typed": raw.get("last_typed") if isinstance(raw.get("last_typed"), str) else "",
    }


def _load(lesson_id: str) -> dict:
    """slide number (as text, the way JSON keys come back) -> record. Missing or corrupt means no history."""
    with _lock:
        try:
            data = json.loads(_quiz_path(lesson_id).read_text(encoding="utf-8"))
            answers = data.get("answers")
        except (OSError, ValueError, AttributeError):
            return {}
    return {str(key): _clean_record(value) for key, value in answers.items()} if isinstance(answers, dict) else {}


def _save(lesson_id: str, answers: dict):
    path = _quiz_path(lesson_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps({"lesson_id": lesson_id, "answers": answers}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a reader never sees half an answer file
    finally:
        tmp.unlink(missing_ok=True)   # only left behind when the write or the replace failed


def _questions(lesson: dict) -> list[tuple[int, dict]]:
    """(1-based slide number, slide) for every check slide that really has a question and an answer."""
    found = []
    for number, slide in enumerate(lesson.get("slides") or [], start=1):
        if (isinstance(slide, dict) and slide.get("kind") == "check"
                and isinstance(slide.get("question"), str) and slide["question"].strip()
                and isinstance(slide.get("answer"), str) and slide["answer"].strip()):
            found.append((number, slide))
    return found


def _history(answers: dict, number: int, slide: dict) -> dict:
    """The saved record for a slide, unless it was written for a different question (the slide
    was reworded since): then the old result says nothing about this one."""
    record = answers.get(str(number))
    return record if record and record["question"] == slide["question"] else _clean_record(None)


def _state(lesson_id: str, lesson: dict, answers: dict) -> dict:
    questions = []
    for number, slide in _questions(lesson):
        record = _history(answers, number, slide)
        questions.append({
            "slide": number, "question": slide["question"], "answer": slide["answer"],
            "attempts": record["attempts"], "got": record["got"], "missed": record["missed"],
            "awaiting_review": record["awaiting_review"], "last_result": record["last_result"],
            "last_typed": record["last_typed"],
        })
    return {"lesson_id": lesson_id, "questions": questions,
            "awaiting_review": sum(1 for q in questions if q["awaiting_review"])}


def get_state(lesson_id: str) -> dict:
    lesson = lessons.get_lesson(_check_id(lesson_id))   # FileNotFoundError for an unknown lesson
    return _state(lesson_id, lesson, _load(lesson_id))


def record_answer(payload: dict) -> dict:
    """Records one self-graded answer and returns the lesson's updated state."""
    lesson_id = payload.get("lesson_id")
    lesson = lessons.get_lesson(_check_id(lesson_id))
    slide_number, result, typed = payload.get("slide"), payload.get("result"), payload.get("typed")
    if result not in RESULTS:
        raise ValueError("The result must be 'got' or 'missed'")
    if typed is None:
        typed = ""
    if not isinstance(typed, str):
        raise ValueError("The typed answer must be text")
    if len(typed) > MAX_TYPED:
        raise ValueError(f"The typed answer is too long ({MAX_TYPED} characters at most)")
    try:
        typed.encode("utf-8")
    except UnicodeEncodeError:      # a lone surrogate from a JSON escape; it could never be written to the file
        raise ValueError("The typed answer has characters that can't be saved") from None
    # type() rather than isinstance: True is an int, and slide 0 would index the last slide
    if type(slide_number) is not int or slide_number < 1:
        raise ValueError("The slide must be a number from 1")
    slide = dict(_questions(lesson)).get(slide_number)
    if slide is None:
        raise ValueError("That slide isn't a quiz question")
    with _lock:
        answers = _load(lesson_id)
        record = _history(answers, slide_number, slide)
        record.update(question=slide["question"], attempts=record["attempts"] + 1,
                      last_result=result, last_typed=typed)
        record[result] += 1
        if result == "missed":
            record.update(awaiting_review=True, last_missed_at=time.time())
        else:
            record["awaiting_review"] = False
        answers[str(slide_number)] = record
        _save(lesson_id, answers)
    return _state(lesson_id, lesson, answers)


def reset(payload: dict) -> dict:
    """Forgets one lesson's results, or every lesson's when asked for all explicitly."""
    if "lesson_id" in payload:
        paths = [_quiz_path(payload["lesson_id"])]      # a lesson deleted since can still have its leftovers cleared
    elif payload.get("all") is True:
        paths = list(_quiz_dir().glob("*.json")) if _quiz_dir().is_dir() else []
    else:
        raise ValueError("Say which lesson to reset, or all of them")
    cleared = 0
    with _lock:
        for path in paths:
            try:
                path.unlink()
                cleared += 1
            except FileNotFoundError:
                pass
    return {"cleared": cleared}


def review() -> dict:
    """Every question, in any lesson, that was missed and not yet answered correctly since -
    the one missed longest ago last. A lesson that has been deleted just drops out."""
    items = []
    for path in sorted(_quiz_dir().glob("*.json")) if _quiz_dir().is_dir() else []:
        lesson_id = path.stem
        answers = _load(lesson_id)
        if not any(record["awaiting_review"] for record in answers.values()):
            continue
        try:
            lesson = lessons.get_lesson(lesson_id)
        except (FileNotFoundError, OSError, ValueError):
            continue
        for number, slide in _questions(lesson):
            record = _history(answers, number, slide)
            if record["awaiting_review"]:
                items.append({"lesson_id": lesson_id, "lesson_title": lesson.get("title") or "",
                              "class_code": lesson.get("class_code") or "", "slide": number,
                              "question": slide["question"], "answer": slide["answer"],
                              "missed": record["missed"], "last_missed_at": record["last_missed_at"]})
    items.sort(key=lambda item: (-item["last_missed_at"], item["lesson_id"], item["slide"]))
    return {"count": len(items), "items": items}
