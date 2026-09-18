"""Regression tests for the pure logic in this app.

Scope is deliberate: everything here runs in seconds with no GPU, microphone, network
or Claude call, so it can run on every change. Transcription accuracy and prompt quality are
not unit-testable and are measured separately by the harnesses in tools/.

Nearly every test below is a real bug that reached real lecture notes or crashed a live
session during development. They exist to keep those specific failures from coming back,
which is why each one names what went wrong rather than just asserting behaviour.

    python -m pytest tests/ -q          (or: python tests/test_regressions.py)
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

import capture
import dashboard
import local_formatter
import notes
import telemetry


# ---------------------------------------------------------------------------
# Assistant chatter reaching the notes file
# ---------------------------------------------------------------------------

class TestAssistantChatterGuard:
    """`claude -p` once decided to act as an agent, tried to write the notes file
    itself, hit a permission gate it couldn't clear headlessly, and returned its
    apology as the answer - which was saved verbatim as a lecture's notes. The whole
    lecture had to be recovered from audio."""

    def test_rejects_the_exact_text_that_corrupted_biol_notes(self):
        real = ("The edit needs your approval to write to `notes/BIOL_1440.md`. Please "
                "grant permission and I'll proceed, or let me know if you'd like changes "
                "to the content first.")
        assert notes._looks_like_assistant_chatter(real)

    def test_rejects_the_study_guide_variant(self):
        real = ("I've drafted the consolidated study guide, organized by topic. It's ready "
                "to write to `notes/BIOL_1440_study_guide.md`. Please grant permission for "
                "the write.")
        assert notes._looks_like_assistant_chatter(real)

    def test_rejects_the_variant_found_sitting_in_wrtg_notes(self):
        """Found during an end-to-end smoke test, already in real WRTG 1310 notes. An
        earlier grep for "approval"/"grant permission" missed it - this one says
        "approve"/"I need permission" - which is the whole reason the phrase list gets
        widened from observed failures rather than assumed complete."""
        real = ("I need permission to write to that file — please approve the write when "
                "prompted, or let me know if you'd like the notes elsewhere.")
        assert notes._looks_like_assistant_chatter(real)

    def test_accepts_normal_notes(self):
        assert not notes._looks_like_assistant_chatter(
            "## Monday, September 1, 2026\n\n### Covalent Bonding\n"
            "- A **covalent bond** is a shared pair of electrons.\n")

    def test_accepts_structured_notes_quoting_a_trigger_phrase(self):
        """A lecturer really can say "let me know if that's unclear" - the guard needs
        both a trigger phrase AND absent markdown structure, or it eats real Q&A."""
        assert not notes._looks_like_assistant_chatter(
            "## Monday\n\n### Q&A\n- **A (Instructor):** Let me know if that's unclear.")


# ---------------------------------------------------------------------------
# Startup menu
# ---------------------------------------------------------------------------

class TestStartupMenu:
    """The menu is the only way most of these features get discovered, and a wrong
    mapping would silently start the wrong thing - recording when you asked for the
    dashboard is a particularly bad failure, since it opens the mic."""

    @pytest.mark.parametrize("keystroke,expected", [
        ("1", "record"),
        ("", "record"),          # bare Enter takes the default
        ("2", "study_guide"),
        ("3", "flashcards"),
        ("4", "dashboard"),
        ("99", "record"),        # unrecognised input must not do something surprising
        ("  3  ", "flashcards"),  # whitespace tolerated
    ])
    def test_menu_choice_maps_to_the_right_action(self, keystroke, expected, monkeypatch):
        import main
        monkeypatch.setattr("builtins.input", lambda *a: keystroke)

        class Args:
            klass = None

        assert main.choose_action(Args()) == expected


# ---------------------------------------------------------------------------
# Duplicate "## <date>" headings
# ---------------------------------------------------------------------------

class TestDateSectionBoundary:
    """Running the app twice in one day produced two "## <date>" headings for the same
    lecture, because the condense pass scoped itself to this process's start offset
    instead of the date's actual first appearance."""

    def test_finds_first_occurrence_of_the_date(self):
        import main
        content = ("# Class (X 100)\n\n## Monday, September 1, 2026\n- first run\n\n"
                   "## Monday, September 1, 2026\n- second run\n")
        boundary = main._find_date_section_boundary(content, "Monday, September 1, 2026")
        assert content[boundary:].startswith("## Monday, September 1, 2026\n- first run")

    def test_returns_end_when_date_absent_so_nothing_is_truncated(self):
        import main
        content = "# Class (X 100)\n\n## Tuesday, September 2, 2026\n- other day\n"
        assert main._find_date_section_boundary(content, "Monday, September 1, 2026") == len(content)

    def test_ignores_a_different_date(self):
        import main
        content = "## Sunday, August 31, 2026\n- a\n\n## Monday, September 1, 2026\n- b\n"
        boundary = main._find_date_section_boundary(content, "Monday, September 1, 2026")
        assert content[boundary:].startswith("## Monday, September 1, 2026")


# ---------------------------------------------------------------------------
# Heuristic formatter headings
# ---------------------------------------------------------------------------

class TestHeuristicHeadings:
    """A mis-transcribed initial ("...we'll talk about A.") was treated as a sentence
    boundary and became a "### A" heading in real psychology notes."""

    def test_rejects_single_letter_heading(self):
        assert local_formatter._extract_heading("all right we will talk about A.") is None

    def test_keeps_real_multiword_heading(self):
        assert local_formatter._extract_heading(
            "now we are going to talk about the amygdala and its role in fear") is not None

    def test_keeps_real_single_long_word_heading(self):
        assert local_formatter._extract_heading("we will talk about aggression") == "Aggression"


# ---------------------------------------------------------------------------
# Flashcard JSON parsing
# ---------------------------------------------------------------------------

class TestFlashcardParsing:
    def test_parses_plain_json_array(self):
        raw = ('[{"question": "Q1?", "options": ["a","b","c","d"], '
               '"correct_index": 2, "explanation": "because"}]')
        cards = notes._parse_flashcards_json(raw)
        assert len(cards) == 1 and cards[0]["correct_index"] == 2

    def test_tolerates_code_fences_and_preamble(self):
        """Models don't reliably obey "output ONLY JSON"."""
        raw = ('Here you go:\n```json\n[{"question": "Q?", "options": ["a","b","c","d"], '
               '"correct_index": 0, "explanation": "e"}]\n```')
        assert len(notes._parse_flashcards_json(raw)) == 1

    def test_drops_malformed_cards_but_keeps_good_ones(self):
        raw = ('[{"question": "ok", "options": ["a","b","c","d"], "correct_index": 1, '
               '"explanation": "e"},'
               ' {"question": "bad - only two options", "options": ["a","b"], '
               '"correct_index": 0, "explanation": "e"},'
               ' {"question": "bad - index out of range", "options": ["a","b","c","d"], '
               '"correct_index": 9, "explanation": "e"}]')
        cards = notes._parse_flashcards_json(raw)
        assert len(cards) == 1 and cards[0]["question"] == "ok"

    def test_returns_none_when_nothing_usable(self):
        assert notes._parse_flashcards_json("I couldn't generate a quiz.") is None


# ---------------------------------------------------------------------------
# Anki export
# ---------------------------------------------------------------------------

class TestAnkiExport:
    def test_writes_one_tab_separated_row_per_card(self, tmp_path, monkeypatch):
        monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
        cards = [{"question": "What is X?", "options": ["right", "w1", "w2", "w3"],
                   "correct_index": 0, "explanation": "because X"}]
        path = notes.export_flashcards_anki("TEST 100", "Test Class", cards)
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [l for l in lines if "\t" in l]
        assert len(rows) == 1
        front, back = rows[0].split("\t")
        assert front == "What is X?"
        assert "right" in back and "because X" in back
        # the distractors must NOT be on the card - recall, not recognition
        assert "w1" not in back

    def test_sanitises_tabs_and_newlines_that_would_break_the_columns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
        cards = [{"question": "line1\nline2\twith tab", "options": ["a", "b", "c", "d"],
                   "correct_index": 0, "explanation": "exp\nwith newline"}]
        path = notes.export_flashcards_anki("TEST 100", "Test Class", cards)
        rows = [l for l in path.read_text(encoding="utf-8").splitlines() if "\t" in l]
        assert len(rows) == 1 and rows[0].count("\t") == 1


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    @pytest.fixture
    def notes_dir(self, tmp_path, monkeypatch):
        import search as search_module
        (tmp_path / "TEST_100.md").write_text(
            "# Test Class (TEST 100)\n\n"
            "## Monday, September 1, 2026\n\n"
            "### Photosynthesis\n"
            "- Plants convert **light** into chemical energy.\n"
            "- Occurs in the chloroplast.\n\n"
            "## Tuesday, September 2, 2026\n\n"
            "### Respiration\n"
            "- Mitochondria release energy.\n", encoding="utf-8")
        monkeypatch.setattr(search_module, "NOTES_DIR", tmp_path)
        monkeypatch.setattr(search_module, "STATE_DIR", tmp_path / "nonexistent")
        return search_module

    def test_finds_term_and_reports_its_lecture_and_section(self, notes_dir):
        hits = notes_dir.search_notes("chloroplast")
        assert len(hits) == 1
        assert hits[0].location == "Monday, September 1, 2026"
        assert hits[0].section == "Photosynthesis"

    def test_is_case_insensitive(self, notes_dir):
        assert notes_dir.search_notes("MITOCHONDRIA")

    def test_multi_word_query_requires_all_terms(self, notes_dir):
        assert notes_dir.search_notes("plants light")
        assert not notes_dir.search_notes("plants mitochondria")

    def test_headings_are_navigation_not_results(self, notes_dir):
        """Matching the "### Respiration" heading itself would return a result with no
        content in it."""
        assert all(not h.text.startswith("#") for h in notes_dir.search_notes("respiration"))

    def test_empty_query_returns_nothing_rather_than_everything(self, notes_dir):
        assert notes_dir.search_notes("   ") == []


# ---------------------------------------------------------------------------
# Dashboard markdown rendering
# ---------------------------------------------------------------------------

class TestMarkdownRendering:
    def test_escapes_html_before_formatting(self):
        """Notes are LLM-generated; nothing in them may be trusted as markup."""
        out = dashboard.markdown_to_html("- <script>alert(1)</script>")
        assert "<script>" not in out and "&lt;script&gt;" in out

    def test_renders_bold_and_italic(self):
        out = dashboard.markdown_to_html("- A **bold** and *italic* line")
        assert "<strong>bold</strong>" in out and "<em>italic</em>" in out

    def test_nested_bullets_produce_nested_lists(self):
        out = dashboard.markdown_to_html("- outer\n  - inner\n")
        assert out.count("<ul>") == 2 and out.count("</ul>") == 2

    def test_date_headings_get_anchors_for_the_table_of_contents(self):
        out = dashboard.markdown_to_html("## Monday, September 1, 2026")
        assert 'id="monday-september-1-2026"' in out

    def test_all_lists_are_closed(self):
        out = dashboard.markdown_to_html("- a\n  - b\n\n## Heading\n\n- c\n")
        assert out.count("<ul>") == out.count("</ul>")


# ---------------------------------------------------------------------------
# Telemetry / command channel
# ---------------------------------------------------------------------------

class TestTelemetry:
    @pytest.fixture
    def tel_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(telemetry, "STATE_DIR", tmp_path)
        monkeypatch.setattr(telemetry, "STATUS_PATH", tmp_path / "status.json")
        monkeypatch.setattr(telemetry, "COMMAND_PATH", tmp_path / "command.json")
        return tmp_path

    # start_session() starts a heartbeat thread. Each test ends its session so the thread
    # can't outlive the temp paths and write fake status into a live recording's file.
    def test_status_round_trips(self, tel_dir):
        tel = telemetry.Telemetry()
        tel.start_session(class_code="TEST 100", class_title="Test")
        try:
            tel.add_transcript("12:00:00", "hello")
            tel.flush()
            status = telemetry.read_status()
            assert status["active"] and status["class_code"] == "TEST 100"
            assert status["transcript"][-1]["text"] == "hello"
        finally:
            tel.end_session()

    def test_realtime_factor_is_gpu_seconds_per_audio_second(self, tel_dir):
        tel = telemetry.Telemetry()
        tel.start_session(class_code="T", class_title="T")
        try:
            tel.record_chunk(audio_seconds=20.0, transcribe_seconds=5.0, segments=2)
            tel.flush()
            assert telemetry.read_status()["realtime_factor"] == pytest.approx(0.25)
        finally:
            tel.end_session()

    def test_command_is_consumed_exactly_once(self, tel_dir):
        """A dashboard click must not be able to fire the same save twice."""
        assert telemetry.send_command("save_now")
        assert telemetry.take_command() == "save_now"
        assert telemetry.take_command() is None

    def test_missing_status_file_reads_as_empty_not_a_crash(self, tel_dir):
        assert telemetry.read_status() == {}


# ---------------------------------------------------------------------------
# Pause detection (kept off, but the maths must stay honest)
# ---------------------------------------------------------------------------

class TestPauseDetection:
    def test_context_equal_to_tail_cannot_report_a_pause(self):
        """The original bug: the comparison window was collected only until it covered
        the tail, making them the same samples and reducing the test to rms < 0.25*rms.
        Pause detection silently never fired at all."""
        blocks = [np.full(8000, 0.05, dtype=np.float32) for _ in range(4)]
        assert capture.ends_on_pause(blocks, tail_frames=4000, context_frames=4000) is False

    def test_quiet_tail_after_loud_speech_is_a_pause(self):
        """The tail here sits at 0.008 - deliberately ABOVE audio.is_silent()'s 0.002
        absolute threshold, matching a real lecture hall's measured noise floor of
        ~0.013. It can therefore only be detected by the ratio against surrounding
        speech, which is the thing the original bug broke. An earlier version of this
        test used a near-silent 0.001 tail and passed under the buggy code too, via the
        absolute threshold - it never exercised the ratio at all."""
        loud = [np.full(8000, 0.05, dtype=np.float32) for _ in range(20)]
        room_noise = [np.full(8000, 0.008, dtype=np.float32)]
        assert capture.ends_on_pause(loud + room_noise, tail_frames=4000, context_frames=96000)

    def test_continuous_speech_is_not_a_pause(self):
        speech = [np.full(8000, 0.05, dtype=np.float32) for _ in range(20)]
        assert not capture.ends_on_pause(speech, tail_frames=4000, context_frames=96000)

    def test_insufficient_history_is_not_a_pause(self):
        blocks = [np.full(8000, 0.05, dtype=np.float32)]
        assert not capture.ends_on_pause(blocks, tail_frames=4000, context_frames=96000)


# ---------------------------------------------------------------------------
# Session thread-safety
# ---------------------------------------------------------------------------

class TestSessionThreadSafety:
    def test_concurrent_writes_and_saves_lose_no_segments(self, tmp_path):
        """Transcription moved to a background thread while autosave still pops from the
        main one, so these genuinely run at the same time. Every segment must appear
        exactly once, in order, across all the popped text."""
        import main
        session = main.Session(tmp_path / "s.wav")
        n = 150
        done = threading.Event()
        popped = []

        def writer():
            for i in range(n):
                session.write_chunk(np.zeros(160, dtype=np.float32),
                                     [{"start": 0.0, "end": 0.1, "text": f"seg-{i}"}])
            done.set()

        def saver():
            while not done.is_set():
                if text := session.pop_pending_text():
                    popped.append(text)
                time.sleep(0.001)
            if text := session.pop_pending_text():
                popped.append(text)

        threads = [threading.Thread(target=writer), threading.Thread(target=saver)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        session._sf.close()

        assert " ".join(popped).split() == [f"seg-{i}" for i in range(n)]
        assert len(session.all_segments) == n


# ---------------------------------------------------------------------------
# Backup retention
# ---------------------------------------------------------------------------

class TestAudioRetention:
    def _make(self, directory, name, age_days):
        path = directory / name
        path.write_bytes(b"x" * 1000)
        when = time.time() - age_days * 86400
        os.utime(path, (when, when))
        return path

    def test_prunes_old_audio_keeps_transcripts_and_recent_audio(self, tmp_path, monkeypatch):
        import main
        monkeypatch.setattr(main, "STATE_DIR", tmp_path)
        monkeypatch.setattr(main, "AUDIO_RETENTION_DAYS", 30)
        self._make(tmp_path, "OLD_20260101_120000.wav", 40)
        self._make(tmp_path, "OLD_20260101_120000_raw.txt", 40)
        self._make(tmp_path, "NEW_20260901_120000.wav", 1)

        main.auto_prune_audio()
        remaining = {p.name for p in tmp_path.iterdir()}
        assert "OLD_20260101_120000.wav" not in remaining
        assert "OLD_20260101_120000_raw.txt" in remaining, "transcripts are kept forever"
        assert "NEW_20260901_120000.wav" in remaining

    def test_zero_disables_pruning(self, tmp_path, monkeypatch):
        import main
        monkeypatch.setattr(main, "STATE_DIR", tmp_path)
        monkeypatch.setattr(main, "AUDIO_RETENTION_DAYS", 0)
        self._make(tmp_path, "ANCIENT_20200101_120000.wav", 9999)
        main.auto_prune_audio()
        assert (tmp_path / "ANCIENT_20200101_120000.wav").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
