"""Lecture highlights and Jev search ranking, with Jev replaced by a keyword fake. Covers what is
stored (probabilities, never text), when a segment is re-asked, where each lecture's box goes,
and that a recording in progress is left alone."""
import json
import time

import pytest

import highlights
import jev
import search

SID = "PSYC_1300_20260923_130155"
MARKERS = {"exam": "on the exam", "not_tested": "don't memorize", "announcement": "due friday",
           "mishearing": "sell membrane"}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(highlights, "STATE_DIR", tmp_path)
    monkeypatch.setattr(highlights, "course_terms", lambda code: ["cell membrane"])
    monkeypatch.setattr(highlights, "live_session", lambda: None)
    monkeypatch.setattr(jev, "SETTINGS_PATH", tmp_path / "jev_settings.json")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test")
    asked = []

    def fake_ask(state, questions, key=None, timeout=None):
        asked.append(state)
        text = state.get("passage", "").lower()
        if "answers" in questions:  # search ranking
            p = 0.9 if state["search"].lower() in text else 0.1
            return {"answers": {"answers": {"noul": p}}, "model": "jev-test", "input_tokens": 50}
        return {"answers": {k: {"type": "noul", "noul": 0.9 if MARKERS[k] in text else 0.05} for k in questions},
                "model": "jev-test", "input_tokens": 100}

    monkeypatch.setattr(jev, "ask", fake_ask)
    return tmp_path, asked


def write_recording(folder, sid, texts, wav=True):
    lines = [json.dumps({"start": 30.0 * i, "end": 30.0 * (i + 1), "text": t}) for i, t in enumerate(texts)]
    (folder / f"{sid}_segments.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if wav:
        (folder / f"{sid}.wav").write_bytes(b"RIFF")


TEXTS = ["Welcome back everyone.",
         "Rods and cones will be on the exam, so know them.",
         "The paper is due Friday at noon.",
         "You don't memorize the wavelengths, just the idea.",
         "Water crosses the sell membrane by osmosis."]


def test_scan_stores_probabilities_and_no_text(lab):
    folder, asked = lab
    write_recording(folder, SID, TEXTS)
    result = highlights.scan_session(SID)
    assert result["complete"] and len(asked) == 5
    assert result["usage"]["calls"] == 5 and result["usage"]["input_tokens"] == 500
    stored = (folder / f"{SID}_highlights.json").read_text(encoding="utf-8")
    for word in ("Welcome", "Rods", "Friday", "wavelengths", "osmosis"):
        assert word not in stored
    rows = json.loads(stored)["segments"]
    assert rows[1]["exam"] == 0.9 and rows[0]["exam"] == 0.05
    # What Jev saw: the course, the passage, and a little context either side. Calls run in
    # parallel, so look them up by passage rather than by arrival order.
    seen = {state["passage"]: state for state in asked}
    assert seen[TEXTS[1]]["course"] == "Introduction to Psychology (PSYC 1300)"
    assert seen[TEXTS[1]]["before"] == TEXTS[0] and seen[TEXTS[1]]["after"] == TEXTS[2]
    assert seen[TEXTS[0]]["before"] == "" and seen[TEXTS[4]]["after"] == ""


def test_items_over_threshold_carry_moment_and_text(lab):
    folder, _ = lab
    write_recording(folder, SID, TEXTS)
    highlights.scan_session(SID)
    items = highlights.session_items(SID)
    assert [i["index"] for i in items["exam"]] == [1]
    assert items["exam"][0]["clock"] == "0:30" and items["exam"][0]["start"] == 30.0
    assert items["exam"][0]["audio_file"] == SID + ".wav" and "Rods" in items["exam"][0]["text"]
    assert [i["index"] for i in items["announcement"]] == [2]
    assert [i["index"] for i in items["not_tested"]] == [3]
    assert highlights.mishearings(SID)["segments"] == [{"index": 4, "p": 0.9}]


def test_rescan_only_asks_what_changed(lab):
    folder, asked = lab
    write_recording(folder, SID, TEXTS)
    highlights.scan_session(SID)
    assert not highlights.needs_scan(SID)
    asked.clear()
    highlights.scan_session(SID)
    assert asked == []
    # A correction to segment 2 changes its own state and its neighbours' before/after.
    from transcripts import load_transcript
    source = load_transcript(folder, SID)
    overlay = {"source_revision": source["source_revision"], "edits": {"2": "The essay is due Friday."}}
    (folder / f"{SID}_corrections.json").write_text(json.dumps(overlay), encoding="utf-8")
    assert highlights.needs_scan(SID)
    result = highlights.scan_session(SID)
    assert sorted(s["passage"] for s in asked) == sorted([TEXTS[1], "The essay is due Friday.", TEXTS[3]])
    assert result["usage"]["calls"] == 8  # running total across both scans


def test_changed_questions_are_asked_again(lab, monkeypatch):
    folder, asked = lab
    write_recording(folder, SID, TEXTS)
    highlights.scan_session(SID)
    monkeypatch.setattr(highlights, "QUESTIONS_VERSION", "edited")
    assert highlights.needs_scan(SID)
    assert highlights.session_items(SID)["exam"] == []  # stale answers are not shown


def test_glossary_changes_do_not_force_a_rescan(lab, monkeypatch):
    folder, _ = lab
    write_recording(folder, SID, TEXTS)
    highlights.scan_session(SID)
    monkeypatch.setattr(highlights, "course_terms", lambda code: ["cell membrane", "retina"])
    assert not highlights.needs_scan(SID)


def test_contact_details_are_redacted_before_sending(lab):
    folder, asked = lab
    write_recording(folder, SID, ["Email me at prof@school.edu about the paper due Friday."])
    highlights.scan_session(SID)
    assert "prof@school.edu" not in json.dumps(asked) and "[email]" in asked[0]["passage"]


def test_rejected_key_leaves_scan_incomplete_and_pauses_scanner(lab, monkeypatch):
    folder, _ = lab
    write_recording(folder, SID, TEXTS)

    def rejected(*a, **k):
        raise jev.JevError("auth", "TypeSafe answered HTTP 401")

    monkeypatch.setattr(jev, "ask", rejected)
    scanner = highlights.Scanner()
    assert scanner.run_once() is True
    stored = highlights.read_result(SID)
    assert not stored["complete"] and stored["last_error"]["kind"] == "auth"
    assert scanner.run_once() is False  # same key: waits instead of retrying
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_new")
    scanner._retry_at.clear()
    assert scanner.run_once() is True


def test_scanner_skips_the_recording_in_progress(lab, monkeypatch):
    folder, asked = lab
    write_recording(folder, SID, TEXTS)
    other = "PSYC_1300_20260921_130203"
    write_recording(folder, other, TEXTS[:2])
    monkeypatch.setattr(highlights, "live_session", lambda: SID)
    scanner = highlights.Scanner()
    assert scanner.pending() == [other]
    assert scanner.run_once() and not scanner.run_once()
    assert highlights.read_result(SID) is None


def test_scanner_does_nothing_without_a_key_or_when_off(lab, monkeypatch):
    folder, asked = lab
    write_recording(folder, SID, TEXTS)
    jev.save_settings(False)
    assert highlights.Scanner().run_once() is False
    jev.save_settings(True)
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setattr(jev, "SWITCHBOARD_ENV", folder / "missing.env")
    assert highlights.Scanner().run_once() is False
    assert asked == []


def test_boxes_go_under_the_first_heading_of_each_date(lab):
    folder, _ = lab
    write_recording(folder, SID, TEXTS)
    second = "PSYC_1300_20260923_133000"  # same day, recorded in two parts
    write_recording(folder, second, ["Quiz is on the exam day, due Friday."])
    unscanned = "PSYC_1300_20260921_130203"
    write_recording(folder, unscanned, TEXTS[:1])
    highlights.scan_session(SID)
    highlights.scan_session(second)
    headings = [("Monday, September 21, 2026", "monday-september-21-2026"),
                ("Wednesday, September 23, 2026", "wednesday-september-23-2026"),
                ("Wednesday, September 23, 2026", "wednesday-september-23-2026-2")]
    data = highlights.for_class("PSYC 1300", headings)
    boxes = {b["anchor"]: b for b in data["lectures"]}
    assert set(boxes) == {"monday-september-21-2026", "wednesday-september-23-2026"}
    today = boxes["wednesday-september-23-2026"]
    assert sorted(today["sessions"]) == [SID, second] and today["pending"] == []
    assert [(i["session"], i["index"], i["part"]) for i in today["announcement"]] == [(SID, 2, 1), (second, 0, 2)]
    assert boxes["monday-september-21-2026"]["pending"] == ["waiting"]
    assert data["jev"]["ready"]
    # Other classes' recordings never appear.
    assert highlights.for_class("BIOL 1440", headings)["lectures"] == []


def test_live_session_reads_the_recorder_status(monkeypatch):
    import telemetry
    status = {"active": True, "updated_at": time.time(),
              "wav_path": r"C:\AI\Projects\lecture-notes\state\PSYC_1300_20260923_130155.wav"}
    monkeypatch.setattr(telemetry, "read_status", lambda: status)
    assert highlights.live_session() == SID
    status["updated_at"] -= 600  # a recorder that stopped reporting is not recording
    assert highlights.live_session() is None
    status.update(active=False, updated_at=time.time())
    assert highlights.live_session() is None


def test_live_recording_is_marked_as_recording(lab, monkeypatch):
    folder, _ = lab
    write_recording(folder, SID, TEXTS)
    monkeypatch.setattr(highlights, "live_session", lambda: SID)
    data = highlights.for_class("PSYC 1300", [("Wednesday, September 23, 2026", "w")])
    assert data["lectures"][0]["pending"] == ["recording"]


def test_legacy_transcript_uses_wall_clock_stamps(lab):
    folder, _ = lab
    sid = "PSYC_1300_20260902_125951"
    (folder / f"{sid}_raw.txt").write_text(
        "[12:59:58] Hello.\n\n[13:00:29] This is on the exam.\n", encoding="utf-8")
    highlights.scan_session(sid)
    item = highlights.session_items(sid)["exam"][0]
    assert item["clock"] == "13:00:29" and item["start"] is None and item["audio_file"] == ""


def test_long_passages_are_clipped():
    states = highlights.segment_states("PSYC 1300", [{"text": "word " * 2000}, {"text": "next"}])
    assert len(states[0]["passage"]) <= highlights.PASSAGE_CHARS
    assert len(states[1]["before"]) <= highlights.BEFORE_CHARS


# ---------------------------------------------------------------------------
# Search ranking
# ---------------------------------------------------------------------------

ROWS = [{"class_code": "BIOL 1440", "section": "Water", "text": "Hydrogen bonds hold water together."},
        {"class_code": "BIOL 1440", "section": "Buffers", "text": "Buffers keep blood pH stable."},
        {"class_code": "BIOL 1440", "section": "Cells", "text": "Cells divide."}]


def test_jev_rank_puts_the_answer_first_and_caches(lab, monkeypatch):
    _, asked = lab
    search._rank_cache.clear()
    ranked, note = search.jev_rank("blood pH", ROWS)
    assert note is None and ranked[0]["section"] == "Buffers" and ranked[0]["jev"] == 0.9
    assert [r["section"] for r in ranked[1:]] == ["Water", "Cells"]  # ties keep meaning order
    seen = {state["passage"]: state for state in asked}
    assert seen[ROWS[0]["text"]]["search"] == "blood pH" and seen[ROWS[0]["text"]]["section"] == "Water"
    asked.clear()
    search.jev_rank("blood pH", ROWS)
    assert asked == []


def test_jev_rank_falls_back_to_meaning_order(lab, monkeypatch):
    search._rank_cache.clear()

    def offline(*a, **k):
        raise jev.JevError("network", "TypeSafe request failed: URLError")

    monkeypatch.setattr(jev, "ask", offline)
    ranked, note = search.jev_rank("blood pH", ROWS)
    assert ranked == ROWS and "URLError" in note
