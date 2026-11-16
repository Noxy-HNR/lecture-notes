"""Note repairs (repair.py), recording compression (archive.py), the wrong-class check
(class_match.py), revision retention, and the Claude retry/skip rules in notes.py."""
import json
import os
import time
from datetime import datetime

import numpy as np
import pytest
import soundfile as sf

import archive
import class_match
import notes
import repair
import storage

TITLE = "# Principles of Biology I (BIOL 1440)\n\n"


def block(sid, body):
    return f"<!-- session:{sid} -->\n{body.strip()}\n<!-- /session:{sid} -->\n\n"


@pytest.fixture
def library(tmp_path, monkeypatch):
    notes_dir, state = tmp_path / "notes", tmp_path / "state"
    notes_dir.mkdir()
    state.mkdir()
    monkeypatch.setattr(notes, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(notes, "notes_path", lambda code: notes_dir / f"{code.replace(' ', '_')}.md")
    monkeypatch.setattr(repair, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(repair, "STATE_DIR", state)
    monkeypatch.setattr(repair, "JOB_PATH", state / "repair_job.json")
    monkeypatch.setattr(repair, "MEMORY_PATH", state / "repair_memory.json")
    monkeypatch.setattr(repair, "recording_active", lambda: False)
    monkeypatch.setattr(class_match, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(notes.gpu_formatter, "format_transcript",
                        lambda *a, **k: pytest.fail("a repair must never run the local model"))
    return notes_dir, state


def write_segments(state, sid, words):
    (state / f"{sid}_segments.jsonl").write_text(
        "".join(json.dumps({"start": i * 5.0, "end": i * 5.0 + 4, "text": w}) + "\n" for i, w in enumerate(words)),
        encoding="utf-8")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_units_finds_sessions_and_older_dated_sections():
    content = (TITLE + "## Tuesday, September 1, 2026\n\n- old point ⚠️ *verify*\n\n"
               + block("BIOL_1440_20261103_082222", "## Tuesday, November 3, 2026\n\n- a")
               + block("BIOL_1440_20261103_125747", "### Meiosis\n- b"))
    units = repair.parse_units(content)
    assert [u["key"] for u in units] == ["day:Tuesday, September 1, 2026", "session:BIOL_1440_20261103_082222",
                                         "session:BIOL_1440_20261103_125747"]
    assert units[2]["day"] == "Tuesday, November 3, 2026", "a later recording belongs to the heading above it"
    for unit in units:
        assert content[unit["start"]:unit["end"]] == unit["text"]


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def test_resolve_problem_rules():
    original = "## Monday, November 2, 2026\n\n" + "- point about cells ⚠️ *verify*\n" * 30
    good = "## Monday, November 2, 2026\n\n" + "- point about cells\n" * 30
    assert repair.resolve_problem(original, good) is None
    assert "markers" in repair.resolve_problem(original, good + "- x ⚠️ verify")
    assert "date heading" in repair.resolve_problem(original, good.replace("November 2", "November 3"))
    assert "words long" in repair.resolve_problem(original, "## Monday, November 2, 2026\n\n- cells")
    assert "message" in repair.resolve_problem(original, "Here are the cleaned notes:\n" + good)


def test_resolve_replaces_only_that_lecture(library, monkeypatch):
    notes_dir, _ = library
    first = "## Tuesday, November 3, 2026\n\n- all unicyclic recessariums use mitosis ⚠️ *verify*\n- keep me"
    second = "### Gametes\n- the sperm delivers its nucleus"
    path = notes_dir / "BIOL_1440.md"
    path.write_text(TITLE + block("BIOL_1440_20261103_082222", first) + block("BIOL_1440_20261103_125747", second),
                    encoding="utf-8")
    prompts = []
    def fake_cli(prompt, timeout=None, **kwargs):
        prompts.append(prompt)
        return "## Tuesday, November 3, 2026\n\n- All unicellular eukaryotes use mitosis\n- keep me"
    monkeypatch.setattr(notes, "_try_claude_cli_format", fake_cli)

    result = repair.resolve("BIOL 1440", "session:BIOL_1440_20261103_082222")
    assert result["ok"], result
    text = path.read_text(encoding="utf-8")
    assert "unicellular eukaryotes" in text and "⚠️" not in text
    assert second.strip() in text and text.count("<!-- session:") == 2
    assert "Principles of Biology I (BIOL 1440)" in prompts[0]


def test_resolve_keeps_notes_when_reply_fails_checks(library, monkeypatch):
    notes_dir, _ = library
    body = "## Tuesday, November 3, 2026\n\n" + "- a point that matters ⚠️ *verify*\n" * 40
    path = notes_dir / "BIOL_1440.md"
    path.write_text(TITLE + block("BIOL_1440_20261103_082222", body), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    calls = []
    monkeypatch.setattr(notes, "_try_claude_cli_format",
                        lambda p, timeout=None, **k: calls.append(1) or "## Tuesday, November 3, 2026\n\n- short")
    result = repair.resolve("BIOL 1440", "session:BIOL_1440_20261103_082222")
    assert not result["ok"] and "words long" in result["message"]
    assert len(calls) == 2, "one retry before giving up"
    assert path.read_text(encoding="utf-8") == before


def test_repairs_refuse_while_recording(library, monkeypatch):
    monkeypatch.setattr(repair, "recording_active", lambda: True)
    with pytest.raises(repair.RepairRefused):
        repair.resolve("BIOL 1440", "session:X_20261103_082222")
    with pytest.raises(repair.RepairRefused):
        repair.start_job({"class_code": "BIOL 1440"})


# ---------------------------------------------------------------------------
# reformat (and notes.py's rule that a rewrite never runs a tier that can't win)
# ---------------------------------------------------------------------------

def rough_notes(notes_dir, state, sid="BIOL_1440_20261020_123548"):
    body = ("## Tuesday, October 20, 2026\n\n" + notes.HEURISTIC_MARKER + " — Claude CLI/API unavailable)*\n\n"
            "- in the current of oxygen which is the way\n")
    (notes_dir / "BIOL_1440.md").write_text(TITLE + block(sid, body), encoding="utf-8")
    write_segments(state, sid, ["In the presence of oxygen cells get the most energy.", "Fermentation works without it."])
    return sid


def test_reformat_rebuilds_from_transcript_with_claude(library, monkeypatch):
    notes_dir, state = library
    sid = rough_notes(notes_dir, state)
    seen = {}
    def fake_cli(prompt, timeout=None, **kwargs):
        seen["prompt"] = prompt
        return "## Tuesday, October 20, 2026\n\n### Fermentation\n- **Fermentation** works without oxygen"
    monkeypatch.setattr(notes, "_try_claude_cli_format", fake_cli)
    result = repair.reformat("BIOL 1440", sid)
    assert result["ok"], result
    text = (notes_dir / "BIOL_1440.md").read_text(encoding="utf-8")
    assert "**Fermentation** works without oxygen" in text and notes.HEURISTIC_MARKER not in text
    assert "Fermentation works without it." in seen["prompt"]


def test_reformat_without_claude_leaves_notes_and_skips_local_tiers(library, monkeypatch):
    notes_dir, state = library
    sid = rough_notes(notes_dir, state)
    before = (notes_dir / "BIOL_1440.md").read_text(encoding="utf-8")
    def down(prompt, timeout=None, **kwargs):
        notes.last_cli_failure = "the Claude CLI exited with code 1: not signed in"
        return None
    monkeypatch.setattr(notes, "_try_claude_cli_format", down)
    monkeypatch.setattr(notes, "_try_claude_api_format", lambda *a, **k: None)
    monkeypatch.setattr(notes, "_local_format", lambda *a, **k: pytest.fail("heuristic tier would be thrown away"))
    result = repair.reformat("BIOL 1440", sid)
    assert not result["ok"] and result["claude_down"]
    assert (notes_dir / "BIOL_1440.md").read_text(encoding="utf-8") == before


def test_format_retries_a_cli_timeout_once(library, monkeypatch):
    notes_dir, _ = library
    calls = []
    def flaky(prompt, timeout=None, **kwargs):
        calls.append(timeout)
        if len(calls) == 1:
            notes.last_cli_failure = "the Claude CLI did not finish within 3 minutes"
            return None
        notes.last_cli_failure = None
        return "## Tuesday, October 20, 2026\n\n- notes"
    monkeypatch.setattr(notes, "_try_claude_cli_format", flaky)
    _, method = notes.format_and_save("BIOL 1440", "Bio", "words " * 50, "Tuesday, October 20, 2026",
                                      combine_proofread=True, session_id="BIOL_1440_20261020_123548")
    assert method == "cli" and len(calls) == 2


def test_format_does_not_retry_a_signed_out_cli(library, monkeypatch):
    calls = []
    def signed_out(prompt, timeout=None, **kwargs):
        calls.append(1)
        notes.last_cli_failure = "the Claude CLI exited with code 1: Failed to authenticate"
        return None
    monkeypatch.setattr(notes, "_try_claude_cli_format", signed_out)
    monkeypatch.setattr(notes, "_try_claude_api_format", lambda *a, **k: None)
    monkeypatch.setattr(notes.gpu_formatter, "format_transcript", lambda *a, **k: None)
    notes.format_and_save("BIOL 1440", "Bio", "words " * 50, "Tuesday, October 20, 2026", combine_proofread=True)
    assert len(calls) == 1


def test_cli_timeout_grows_with_the_transcript():
    assert notes.cli_timeout_for("x") == notes.CLI_TIMEOUT_SECONDS
    assert notes.cli_timeout_for("word " * 8000) == notes.CLI_TIMEOUT_SECONDS + 160


# ---------------------------------------------------------------------------
# the background job
# ---------------------------------------------------------------------------

def test_job_runs_items_and_stops_when_claude_is_down(library, monkeypatch):
    notes_dir, state = library
    a, b = "BIOL_1440_20261103_082222", "BIOL_1440_20261105_121559"
    (notes_dir / "BIOL_1440.md").write_text(
        TITLE + block(a, "## Tuesday, November 3, 2026\n\n- x ⚠️ *verify*")
        + block(b, "## Thursday, November 5, 2026\n\n- y ⚠️ *verify*"), encoding="utf-8")
    items = repair.plan_items("BIOL 1440")
    assert [i["key"] for i in items] == ["session:" + a, "session:" + b]
    repair._write_json(repair.JOB_PATH, {"id": "j1", "status": "running", "trigger": "auto",
                                         "created_at": time.time(), "items": items})
    monkeypatch.setattr(repair, "_housekeeping", lambda: None)
    monkeypatch.setattr(repair, "resolve", lambda code, key: {"ok": False, "claude_down": True,
                                                              "message": "Claude couldn't be reached"})
    assert repair.run_job("j1") == 0
    job = json.loads(repair.JOB_PATH.read_text(encoding="utf-8"))
    assert job["status"] == "finished"
    assert [i["state"] for i in job["items"]] == ["failed", "skipped"]
    assert "isn't reachable" in job["message"]


def test_a_failed_text_is_not_retried_automatically_until_it_changes(library, monkeypatch):
    notes_dir, _ = library
    sid = "BIOL_1440_20261103_082222"
    (notes_dir / "BIOL_1440.md").write_text(TITLE + block(sid, "## Tuesday, November 3, 2026\n\n- x ⚠️ *verify*"),
                                            encoding="utf-8")
    monkeypatch.setattr(repair, "_housekeeping", lambda: None)
    calls = []
    monkeypatch.setattr(repair, "resolve", lambda code, key: calls.append(key) or {"ok": False, "message": "no"})
    for job_id in ("j1", "j2"):
        repair._write_json(repair.JOB_PATH, {"id": job_id, "status": "running", "trigger": "auto",
                                             "created_at": time.time(), "items": repair.plan_items("BIOL 1440")})
        repair.run_job(job_id)
    assert len(calls) == 1


def test_auto_items_reformat_rough_lectures_everywhere(library):
    notes_dir, state = library
    sid = rough_notes(notes_dir, state)
    items = repair.auto_items()
    assert [(i["key"], i["action"]) for i in items] == [("session:" + sid, "reformat")]


# ---------------------------------------------------------------------------
# moving a lecture to another class
# ---------------------------------------------------------------------------

def test_move_session_moves_notes_and_files(library):
    notes_dir, state = library
    sid = "WRTG_1310_20260904_174944"
    (notes_dir / "WRTG_1310.md").write_text(
        "# Intro to College Writing (WRTG 1310)\n\n"
        + block("WRTG_1310_20260902_110437", "## Wednesday, September 2, 2026\n\n- thesis")
        + block(sid, "## Friday, September 4, 2026\n\n- hexose sugars: C6H12O6"), encoding="utf-8")
    (notes_dir / "BIOL_1440.md").write_text(
        TITLE + block("BIOL_1440_20260903_125816", "## Thursday, September 3, 2026\n\n- atoms")
        + block("BIOL_1440_20260908_121436", "## Tuesday, September 8, 2026\n\n- water"), encoding="utf-8")
    for suffix in ("_raw.txt", ".ogg"):
        (state / f"{sid}{suffix}").write_text("x", encoding="utf-8")
    (state / f"{sid}_highlights.json").write_text(json.dumps({"session": sid}), encoding="utf-8")

    result = repair.move_session({"class_code": "WRTG 1310", "session_id": sid, "target": "BIOL 1440"})
    new_id = "BIOL_1440_20260904_174944"
    assert result["session_id"] == new_id
    biol = (notes_dir / "BIOL_1440.md").read_text(encoding="utf-8")
    assert biol.index("September 3") < biol.index("hexose") < biol.index("September 8"), "date order"
    assert f"<!-- session:{new_id} -->" in biol
    assert "hexose" not in (notes_dir / "WRTG_1310.md").read_text(encoding="utf-8")
    assert (state / f"{new_id}_raw.txt").exists() and (state / f"{new_id}.ogg").exists()
    assert not (state / f"{sid}_raw.txt").exists()
    assert json.loads((state / f"{new_id}_highlights.json").read_text(encoding="utf-8"))["session"] == new_id


def test_move_hands_the_day_heading_to_the_next_recording_that_day(library):
    notes_dir, _ = library
    first, second = "WRTG_1310_20260902_110437", "WRTG_1310_20260902_112335"
    (notes_dir / "WRTG_1310.md").write_text(
        "# Writing (WRTG 1310)\n\n" + block(first, "## Wednesday, September 2, 2026\n\n- moved")
        + block(second, "### Planning\n- stays"), encoding="utf-8")
    (notes_dir / "BIOL_1440.md").write_text(TITLE, encoding="utf-8")
    repair.move_session({"class_code": "WRTG 1310", "session_id": first, "target": "BIOL 1440"})
    wrtg = (notes_dir / "WRTG_1310.md").read_text(encoding="utf-8")
    assert "## Wednesday, September 2, 2026" in wrtg and wrtg.index("September 2") < wrtg.index("stays")


def test_move_refuses_to_overwrite(library):
    notes_dir, state = library
    sid = "WRTG_1310_20260904_174944"
    (notes_dir / "WRTG_1310.md").write_text("# W (WRTG 1310)\n\n" + block(sid, "## Friday, September 4, 2026\n\n- a"),
                                            encoding="utf-8")
    (notes_dir / "BIOL_1440.md").write_text(TITLE, encoding="utf-8")
    (state / f"{sid}_raw.txt").write_text("x", encoding="utf-8")
    (state / "BIOL_1440_20260904_174944_raw.txt").write_text("y", encoding="utf-8")
    with pytest.raises(repair.RepairRefused):
        repair.move_session({"class_code": "WRTG 1310", "session_id": sid, "target": "BIOL 1440"})
    assert "- a" in (notes_dir / "WRTG_1310.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# wrong-class check
# ---------------------------------------------------------------------------

def test_class_match_flags_a_clear_mismatch_only(library):
    notes_dir, _ = library
    bio = "- " + " ".join(["glucose hexose carbohydrate polymer cellulose glycosidic enzyme membrane"] * 160)
    wrtg = "- " + " ".join(["thesis paragraph audience rhetorical essay drafting revision outline"] * 160)
    (notes_dir / "BIOL_1440.md").write_text(TITLE + bio, encoding="utf-8")
    (notes_dir / "WRTG_1310.md").write_text("# Writing (WRTG 1310)\n\n" + wrtg, encoding="utf-8")
    (notes_dir / "CHEM_1450.md").write_text("# Chem (CHEM 1450)\n\n- " + " ".join(["molarity titration"] * 600),
                                            encoding="utf-8")
    lecture = "the most common hexose is glucose and cellulose is a polymer joined by glycosidic links " * 5
    found = class_match.suggest(lecture, "WRTG 1310")
    assert found and found["code"] == "BIOL 1440"
    assert class_match.suggest(lecture, "BIOL 1440") is None
    assert class_match.suggest("glucose hexose", "WRTG 1310") is None, "too short to judge"


# ---------------------------------------------------------------------------
# recordings
# ---------------------------------------------------------------------------

def make_wav(path, seconds=3.0, age_hours=48):
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000, subtype="FLOAT")
    when = time.time() - age_hours * 3600
    os.utime(path, (when, when))


def test_compress_finished_replaces_old_wavs_with_ogg(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "STATE_DIR", tmp_path)
    monkeypatch.setattr(archive, "_active_recording", lambda: None)
    make_wav(tmp_path / "OLD_20261101_090000.wav")
    make_wav(tmp_path / "NEW_20261115_080000.wav", age_hours=1)
    result = archive.compress_finished(log=lambda *a: None)
    assert result["compressed"] == 1 and result["failed"] == 0
    ogg = tmp_path / "OLD_20261101_090000.ogg"
    assert ogg.exists() and not (tmp_path / "OLD_20261101_090000.wav").exists()
    assert abs(sf.info(str(ogg)).duration - 3.0) < 0.2 and sf.info(str(ogg)).samplerate == 16000
    assert time.time() - ogg.stat().st_mtime > 40 * 3600, "keeps the lecture's date for retention"
    assert (tmp_path / "NEW_20261115_080000.wav").exists(), "the day of the lecture stays uncompressed"


def test_nothing_is_compressed_while_a_lecture_is_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "STATE_DIR", tmp_path)
    monkeypatch.setattr(archive, "_active_recording", lambda: "LIVE_20261115_090000")
    make_wav(tmp_path / "OLD_20261101_090000.wav")
    assert archive.compress_finished(log=lambda *a: None)["compressed"] == 0
    assert (tmp_path / "OLD_20261101_090000.wav").exists()


def test_compress_failure_keeps_the_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "STATE_DIR", tmp_path)
    monkeypatch.setattr(archive, "_active_recording", lambda: None)
    broken = tmp_path / "BAD_20261101_090000.wav"
    broken.write_bytes(b"not audio")
    os.utime(broken, (time.time() - 99999, time.time() - 99999))
    assert archive.compress_finished(log=lambda *a: None)["failed"] == 1
    assert broken.exists() and not list(tmp_path.glob("*.ogg*"))


def test_resume_finds_a_compressed_recording(tmp_path):
    import main
    (tmp_path / "BIOL_1440_20261103_125747.ogg").write_bytes(b"x")
    raw, audio_path, code, day = main._resume_backup_paths(tmp_path / "BIOL_1440_20261103_125747_raw.txt")
    assert audio_path.suffix == ".ogg" and code == "BIOL 1440" and day == "Tuesday, November 3, 2026"


def test_dashboard_lists_and_serves_compressed_recordings(tmp_path, monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard, "STATE_DIR", tmp_path)
    make_wav(tmp_path / "BIOL_1440_20261101_090000.wav", seconds=30)
    archive.compress(tmp_path / "BIOL_1440_20261101_090000.wav")
    [row] = dashboard.list_sessions()
    assert row["wav"] == "BIOL_1440_20261101_090000.ogg" and row["compressed"] and 0.4 < row["minutes"] < 0.6


# ---------------------------------------------------------------------------
# revision retention and test isolation
# ---------------------------------------------------------------------------

def test_revision_retention_keeps_recent_and_one_per_day():
    today = datetime(2026, 11, 15)
    names = [f"202611{d:02d}_{h:02d}0000_000000.txt" for d in range(1, 15) for h in range(8, 12)]  # 56 recent
    names += [f"202607{d:02d}_120000_000000.txt" for d in range(1, 4)]                           # >60 days old
    dropped = set(storage.revisions_to_prune(names, today))
    kept = set(names) - dropped
    assert len(kept) == storage.KEEP_RECENT_REVISIONS + 6  # newest 30 reach into Nov 7; Nov 1-6 keep one each
    for day in range(1, 15):
        assert any(n.startswith(f"202611{day:02d}") for n in kept)
    assert not any(n.startswith("202607") for n in kept)


def test_atomic_write_prunes_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "KEEP_RECENT_REVISIONS", 3)
    path = tmp_path / "n.md"
    for i in range(8):
        storage.atomic_write(path, f"version {i}")
    assert len(list((tmp_path / ".revisions" / "n.md").glob("*.txt"))) == 3  # all from today: the 3 newest


def test_startup_failure_log_is_not_the_real_one():
    import main
    real = main.Path(main.__file__).resolve().parent.parent / "state" / "startup_failures.log"
    assert main.STARTUP_FAILURE_LOG.resolve() != real.resolve()
