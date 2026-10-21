"""Saved notes must never get worse than what was already there: a bad vocab.json can't stop a
recording, the glossary can't swap real words, a cut-off API reply isn't saved, a heuristic
rewrite can't replace model-written notes, and a condense reply that lost content is refused."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import local_formatter
import notes
import spelling
import storage
import vocab


# -- vocab.json ------------------------------------------------------------------------------

@pytest.mark.parametrize("content", ['{"BIOL 1440": ["mitosis",]}', '["not", "a", "map"]', "�\x00"])
def test_a_broken_vocab_file_turns_the_glossary_off_instead_of_failing(tmp_path, monkeypatch, content):
    path = tmp_path / "vocab.json"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(vocab, "VOCAB_PATH", path)
    assert vocab.terms_for_class("BIOL 1440") == []
    assert local_formatter.correct_with_glossary("The amigdala reacts.", "BIOL 1440") == "The amigdala reacts."


def test_vocab_entries_that_are_not_term_lists_are_skipped(tmp_path, monkeypatch):
    path = tmp_path / "vocab.json"
    path.write_text('{"_global": "oops", "BIOL 1440": ["mitosis", 7, "", "meiosis"], "CHEM 1450": null}',
                    encoding="utf-8")
    monkeypatch.setattr(vocab, "VOCAB_PATH", path)
    assert vocab.terms_for_class("BIOL 1440") == ["mitosis", "meiosis"]
    import highlights
    assert highlights.course_terms("CHEM 1450") == []
    assert highlights.course_terms("BIOL 1440") == ["mitosis", "meiosis"]


def test_a_vocab_file_saved_with_a_byte_order_mark_still_works(tmp_path, monkeypatch):
    path = tmp_path / "vocab.json"
    path.write_text('{"BIOL 1440": ["mitosis"]}', encoding="utf-8-sig")
    monkeypatch.setattr(vocab, "VOCAB_PATH", path)
    assert vocab.terms_for_class("BIOL 1440") == ["mitosis"]
    vocab.save_learned_terms("BIOL 1440", {"meiosis"})
    assert vocab.terms_for_class("BIOL 1440") == ["mitosis", "meiosis"]


# -- glossary fix-up -------------------------------------------------------------------------

needs_spell_checker = pytest.mark.skipif(spelling.is_english_word("amygdala") is None,
                                         reason="Windows spell checker not available")


@pytest.fixture
def glossary(monkeypatch):
    terms = ["cation", "hydration", "molecule", "vesicle", "genus", "mole", "amygdala",
             "serotonin", "hippocampus", "Rosenthal"]
    monkeypatch.setattr(local_formatter.vocab_module, "terms_for_class", lambda code: terms)


@needs_spell_checker
@pytest.mark.parametrize("word", ["action", "dehydration", "molecules", "vehicle", "genius", "model"])
def test_real_words_are_never_swapped_for_a_glossary_term(glossary, word):
    text = f"The {word} was discussed."
    assert local_formatter.correct_with_glossary(text, "TEST") == text


@needs_spell_checker
def test_misspelled_terms_are_still_fixed_with_the_glossary_casing(glossary):
    fixed = local_formatter.correct_with_glossary(
        "The amigdala and Hippocampas release seratonin, said Rosenthol.", "TEST")
    assert fixed == "The amygdala and Hippocampus release serotonin, said Rosenthal."


def test_without_a_spell_checker_nothing_is_changed(glossary, monkeypatch):
    monkeypatch.setattr(local_formatter.spelling, "is_english_word", lambda word: None)
    assert local_formatter.correct_with_glossary("The amigdala reacts.", "TEST") == "The amigdala reacts."


# -- Anthropic API fallback ------------------------------------------------------------------

def _fake_anthropic(monkeypatch, stop_reason, text="## Monday\n\n- notes"):
    block = types.SimpleNamespace(type="text", text=text)
    response = types.SimpleNamespace(stop_reason=stop_reason, content=[block])
    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.messages = types.SimpleNamespace(create=lambda **kw: (calls.append(kw), response)[1])

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return calls


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_a_cut_off_or_declined_api_reply_is_not_used(monkeypatch, stop_reason):
    _fake_anthropic(monkeypatch, stop_reason)
    assert notes._try_claude_api_format("prompt") is None


def test_a_complete_api_reply_is_used_with_room_for_a_long_lecture(monkeypatch):
    calls = _fake_anthropic(monkeypatch, "end_turn")
    assert notes._try_claude_api_format("prompt") == "## Monday\n\n- notes"
    assert calls[0]["max_tokens"] == notes.API_MAX_TOKENS == 16000


# -- whole-lecture rewrite -------------------------------------------------------------------

@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(notes.docx_export, "rebuild", lambda *a: None)
    monkeypatch.setattr(notes.gpu_formatter, "format_transcript", lambda *a, **k: None)  # GPU rewrite fails
    path = notes.notes_path("TEST")
    path.write_text(storage.update_session("# Test (TEST)\n", "s1", "## Monday\n\n- Model-written notes"),
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("written_by", ["gpu", "cli", "api"])
def test_a_heuristic_rewrite_keeps_model_written_notes(notes_dir, written_by):
    before = notes_dir.read_text(encoding="utf-8")
    _, method = notes.format_and_save("TEST", "Test", "the whole lecture again", "Monday", mode="local",
                                      session_id="s1", replace_session=True, existing_method=written_by)
    assert method == notes.KEPT_EXISTING
    assert notes_dir.read_text(encoding="utf-8") == before


def test_a_local_model_rewrite_keeps_notes_claude_wrote(notes_dir, monkeypatch):
    monkeypatch.setattr(notes.gpu_formatter, "format_transcript", lambda *a, **k: "## Monday\n\n- local model notes")
    _, method = notes.format_and_save("TEST", "Test", "the whole lecture again", "Monday", mode="local",
                                      session_id="s1", replace_session=True, existing_method="cli")
    assert method == notes.KEPT_EXISTING


def test_a_rewrite_at_the_same_tier_replaces_the_session(notes_dir):
    # All-heuristic autosaves: the whole-lecture heuristic rewrite (e.g. with Q&A) still goes in.
    _, method = notes.format_and_save("TEST", "Test", "the whole lecture again", "Monday", mode="local",
                                      session_id="s1", replace_session=True, existing_method="local")
    assert method == "local"
    assert "the whole lecture again" in notes_dir.read_text(encoding="utf-8")


def test_a_recovery_still_replaces_the_session(notes_dir):
    _, method = notes.format_and_save("TEST", "Test", "the recovered lecture", "Monday", mode="local",
                                      session_id="s1", replace_session=True)
    assert method == "local"
    assert "Model-written notes" not in notes_dir.read_text(encoding="utf-8")


# -- condense pass ---------------------------------------------------------------------------

LECTURE = "## Monday, October 20, 2026\n\n" + "\n".join(f"- point {i} about cell membranes and transport" for i in range(30))


@pytest.fixture
def condense_file(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(notes.docx_export, "rebuild", lambda *a: None)
    path = notes.notes_path("TEST")
    path.write_text(storage.update_session("# Test (TEST)\n", "s1", LECTURE), encoding="utf-8")
    return path


@pytest.mark.parametrize("reply", [
    "\n".join(f"- point {i} about cell membranes and transport" for i in range(30)),   # lost the date heading
    "## Monday, October 20, 2026\n\n- membranes matter",                              # lost most content
    "## Monday, October 20, 2026\n\n" + LECTURE.split("\n\n", 1)[1] + "\n\n## Tuesday\n\n- extra",  # two days
])
def test_a_condense_reply_that_lost_content_is_refused(condense_file, monkeypatch, reply):
    before = condense_file.read_text(encoding="utf-8")
    monkeypatch.setattr(notes, "_try_claude_cli_format", lambda prompt: reply)
    ok, reason = notes.condense_session("TEST", "Test", 0, session_id="s1")
    assert not ok and reason.startswith("kept the notes as they were")
    assert condense_file.read_text(encoding="utf-8") == before


def test_a_condense_reply_that_opens_with_a_message_is_refused(condense_file, monkeypatch):
    reply = "No date appears in the source notes, so I won't invent one.\n\n" + LECTURE
    monkeypatch.setattr(notes, "_try_claude_cli_format", lambda prompt: reply)
    ok, reason = notes.condense_session("TEST", "Test", 0, session_id="s1")
    assert not ok and "message" in reason


@pytest.fixture
def second_session_same_day(tmp_path, monkeypatch):
    """A later recording on the same day: its repeated date heading was removed by dedupe."""
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(notes.docx_export, "rebuild", lambda *a: None)
    text = storage.update_session("# Test (TEST)\n", "s1", LECTURE)
    body = "\n".join(f"- afternoon point {i} about enzymes and kinetics" for i in range(30))
    text = storage.update_session(text, "s2", body)
    path = notes.notes_path("TEST")
    path.write_text(text, encoding="utf-8")
    return path


def test_a_later_same_day_session_gets_the_real_date_and_refuses_an_invented_one(second_session_same_day, monkeypatch):
    prompts = []
    body = "\n".join(f"- afternoon point {i} about enzymes and kinetics" for i in range(30))
    monkeypatch.setattr(notes, "_try_claude_cli_format",
                        lambda prompt: (prompts.append(prompt), "## [Lecture date missing]\n\n" + body)[1])
    ok, reason = notes.condense_session("TEST", "Test", 0, session_id="s2")
    assert not ok and "date heading" in reason
    assert '"## Monday, October 20, 2026"' in prompts[0]

    monkeypatch.setattr(notes, "_try_claude_cli_format",
                        lambda prompt: "## Monday, October 20, 2026\n\n" + body)
    ok, _ = notes.condense_session("TEST", "Test", 0, session_id="s2")
    assert ok
    assert second_session_same_day.read_text(encoding="utf-8").count("## Monday, October 20, 2026") == 1


def test_condensing_rough_heuristic_bullets_may_shrink_them(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(notes.docx_export, "rebuild", lambda *a: None)
    rough = ("## Monday, October 20, 2026\n\n*(auto-formatted locally — Claude CLI/API unavailable)*\n\n"
             + "\n".join(f"- so um basically the membrane is like a barrier you know number {i}" for i in range(30)))
    path = notes.notes_path("TEST")
    path.write_text(storage.update_session("# Test (TEST)\n", "s1", rough), encoding="utf-8")
    monkeypatch.setattr(notes, "_try_claude_cli_format",
                        lambda prompt: "## Monday, October 20, 2026\n\n- **Cell membrane**: a selective barrier")
    ok, _ = notes.condense_session("TEST", "Test", 0, session_id="s1")
    assert ok


def test_a_good_condense_reply_is_saved(condense_file, monkeypatch):
    reply = "## Monday, October 20, 2026\n\n" + "\n".join(f"- point {i} about membranes and transport" for i in range(28))
    monkeypatch.setattr(notes, "_try_claude_cli_format", lambda prompt: reply)
    ok, _ = notes.condense_session("TEST", "Test", 0, session_id="s1")
    assert ok
    assert "- point 27 about membranes and transport" in condense_file.read_text(encoding="utf-8")
