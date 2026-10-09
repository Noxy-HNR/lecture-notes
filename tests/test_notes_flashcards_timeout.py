"""generate_flashcards can raise the CLI timeout for a semester of notes, without changing
what every other caller gets by default. The CLI and API are faked; nothing here calls a model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import notes

QUESTION_JSON = ('[{"question": "Q?", "options": ["a", "b", "c", "d"], '
                 '"correct_index": 1, "explanation": "Because b."}]')


def _fake_cli(monkeypatch, tmp_path):
    """Gives CHEM a notes file and records the keyword arguments the CLI call receives."""
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    (tmp_path / "CHEM_1450.md").write_text("# College Chemistry I\n\n- Electrons fill orbitals.\n", encoding="utf-8")
    calls = []

    def fake_cli(prompt, timeout=notes.CLI_TIMEOUT_SECONDS):
        calls.append(timeout)
        return QUESTION_JSON

    monkeypatch.setattr(notes, "_try_claude_cli_format", fake_cli)
    monkeypatch.setattr(notes, "_try_claude_api_format", lambda prompt: None)
    return calls


def test_default_timeout_is_unchanged(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, tmp_path)
    ok, reason, questions = notes.generate_flashcards("CHEM 1450", "College Chemistry I")
    assert ok and reason is None and len(questions) == 1
    assert calls == [notes.CLI_TIMEOUT_SECONDS]


def test_timeout_is_passed_through_to_the_cli(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, tmp_path)
    ok, _, questions = notes.generate_flashcards("CHEM 1450", "College Chemistry I", timeout=900)
    assert ok and questions[0]["correct_index"] == 1
    assert calls == [900]


def test_a_cli_timeout_is_reported_with_the_longer_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    (tmp_path / "CHEM_1450.md").write_text("- Electrons fill orbitals.", encoding="utf-8")
    monkeypatch.setattr(notes, "_find_claude_cli", lambda: "claude")

    def timed_out(*args, **kwargs):
        raise notes.subprocess.TimeoutExpired(cmd="claude", timeout=kwargs["timeout"])

    monkeypatch.setattr(notes.subprocess, "run", timed_out)
    monkeypatch.setattr(notes, "_try_claude_api_format", lambda prompt: None)
    ok, reason, questions = notes.generate_flashcards("CHEM 1450", "College Chemistry I", timeout=600)
    assert not ok and questions is None
    assert notes.last_cli_failure == "the Claude CLI did not finish within 10 minutes"
