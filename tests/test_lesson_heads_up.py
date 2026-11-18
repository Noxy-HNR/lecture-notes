"""Lessons don't carry the notes' "verify" flags over, and keep heads-ups rare and about the subject."""
import json

import lessons


def slide(title, heads_up=""):
    return {"kind": "concept", "title": title, "narration": f"About {title}.", "heads_up": heads_up}


def plan(*slides):
    return json.dumps({"title": "L", "tldr": "t", "slides": list(slides)})


def test_verify_flags_never_reach_the_lesson_prompt():
    notes = ("- If all boys like spinach, Jackson must like spinach ⚠️ *verify* (transcript says \"fight\").\n"
             "- The amygdala [FLAG: possible transcription error - please verify] handles fear ⚠️verify\n")
    prompt = lessons.build_lesson_prompt("Psychology", "PSYC 1300", "2026-11-09", "2026-11-16", notes)
    notes_part = prompt.split("NOTES (")[-1]
    assert "⚠" not in notes_part and "[FLAG" not in notes_part and "verify" not in notes_part.lower()
    assert "Jackson must like spinach" in notes_part and "handles fear" in notes_part


def test_the_prompt_says_to_teach_what_the_professor_taught():
    prompt = lessons.build_lesson_prompt("Psychology", "PSYC 1300", "2026-11-09", "2026-11-09", "- notes")
    assert "What the professor taught is what the exam expects" in prompt
    assert f"at most {lessons.MAX_HEADS_UP} slides" in prompt
    assert "Never mention the transcript" in prompt


def test_heads_ups_about_the_transcript_are_dropped():
    lesson = lessons.parse_lesson(plan(
        slide("A", "The transcript said 'fight', but it means 'like'."),
        slide("B", "The notes had a ⚠️ verify flag here."),
        slide("C", "Students often confuse negative reinforcement with punishment."),
    ))
    assert [s["heads_up"] for s in lesson["slides"]] == \
        ["", "", "Students often confuse negative reinforcement with punishment."]


def test_only_a_few_heads_ups_are_kept():
    lesson = lessons.parse_lesson(plan(*[slide(f"S{i}", f"Common mistake number {i}.") for i in range(10)]))
    kept = [s["heads_up"] for s in lesson["slides"] if s["heads_up"]]
    assert kept == [f"Common mistake number {i}." for i in range(lessons.MAX_HEADS_UP)]
    assert len(lesson["slides"]) == 10, "the slides themselves are all kept"
