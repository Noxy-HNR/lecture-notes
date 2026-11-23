"""The Lessons page's model choice: Claude Sonnet/Opus/Haiku 5.5 through the Claude CLI, or GPT
models through the Codex CLI, each with low/medium/high effort. Fast mode is offered for the
Codex models only: a Claude lesson never runs in Claude's fast mode (it bills extra usage)."""
import json
import subprocess
from pathlib import Path

import pytest

import codex_cli
import lessons
import notes
from test_lessons import NOTES, PLAN, fakes, library  # noqa: F401  (shared fixtures)

TOPIC = "how noise-cancelling headphones work, from scratch"


@pytest.fixture(autouse=True)
def default_model():
    """configure_model changes module state for the whole process: put it back after each test."""
    yield
    lessons.configure_model(None, None, False)


@pytest.fixture
def spawned(library, monkeypatch):
    started = []
    monkeypatch.setattr(lessons, "_spawn_builder", lambda path: started.append(path) or None)
    monkeypatch.setattr(codex_cli, "available", lambda: (True, ""))
    return started


def job_of(spawned):
    return json.loads(spawned[-1].read_text(encoding="utf-8"))


# --- what a build is started with ----------------------------------------------------------------

@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5",
                                   "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-terra", "gpt-6-luna"])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_every_model_takes_every_effort(spawned, model, effort):
    lessons.start_lesson({"topic": TOPIC, "model": model, "effort": effort, "fast": True})
    job = job_of(spawned)
    fast_allowed = model.startswith("gpt-")
    assert (job["model"], job["effort"], job["fast"]) == (model, effort, fast_allowed)
    spawned[-1].unlink()


def test_unknown_choices_fall_back_to_sonnet_high_and_fast_off(spawned):
    lessons.start_lesson({"topic": TOPIC, "model": "gpt-99", "effort": "xhigh", "fast": "yes"})
    job = job_of(spawned)
    assert (job["model"], job["effort"], job["fast"]) == ("claude-sonnet-5-5", "high", False)


def test_an_older_page_without_model_choice_still_builds_with_its_claude_effort(spawned):
    lessons.start_lesson({"topic": TOPIC, "claude_effort": "medium"})
    job = job_of(spawned)
    assert (job["model"], job["effort"], job["fast"]) == ("claude-sonnet-5-5", "medium", False)


def test_a_codex_model_without_codex_installed_is_refused_before_any_job(library, monkeypatch):
    monkeypatch.setattr(lessons, "_spawn_builder", lambda path: pytest.fail("no job should start"))
    monkeypatch.setattr(codex_cli, "available", lambda: (False, "the Codex CLI isn't installed"))
    with pytest.raises(ValueError, match="GPT-6 Luna needs the Codex CLI isn't installed"):
        lessons.start_lesson({"topic": TOPIC, "model": "gpt-6-luna"})


# --- where each call goes ------------------------------------------------------------------------------

def test_claude_choice_sets_model_and_effort_and_never_fast_mode(monkeypatch):
    lessons.configure_model("claude-opus-5-5", "low", True)   # an old page could still send fast
    seen = {}

    def run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, "## Lesson\n\nwritten", "")
    monkeypatch.setattr(notes, "_find_claude_cli", lambda: "claude")
    monkeypatch.setattr(notes.subprocess, "run", run)
    assert lessons._ask_model("prompt", 60) == "## Lesson\n\nwritten"
    args = seen["args"]
    assert args[args.index("--model") + 1] == "claude-opus-5-5"
    assert args[args.index("--effort") + 1] == "low"
    assert "--settings" not in args and "fastMode" not in " ".join(args)
    assert lessons.lesson_fast is False and notes.claude_fast is False


def test_claude_without_fast_mode_passes_no_settings(monkeypatch):
    lessons.configure_model("claude-haiku-5-5", "medium", False)
    seen = {}
    monkeypatch.setattr(notes, "_find_claude_cli", lambda: "claude")
    monkeypatch.setattr(notes.subprocess, "run", lambda args, **k: seen.update(args=args) or
                        subprocess.CompletedProcess(args, 0, "## Lesson\n\nok", ""))
    lessons._ask_model("prompt", 60)
    assert "--settings" not in seen["args"] and seen["args"][seen["args"].index("--model") + 1] == "claude-haiku-5-5"


def test_note_saves_are_untouched_by_a_lesson_choice():
    """A lesson build configures its own process only; the recorder's note saves never change."""
    lessons.configure_model(None, None, False)
    assert notes.claude_model == notes.CLAUDE_MODEL == "claude-sonnet-5-5"
    assert notes.claude_effort == "high" and notes.claude_fast is False


def test_codex_choice_goes_to_codex_with_its_settings(monkeypatch):
    lessons.configure_model("gpt-6.2-sol", "medium", True)
    calls = []
    monkeypatch.setattr(codex_cli, "run", lambda prompt, model, effort, fast, timeout, web_search=False:
                        calls.append((model, effort, fast, web_search)) or '{"ok": true}')
    monkeypatch.setattr(notes, "_try_claude_cli_format", lambda *a, **k: pytest.fail("Claude was called"))
    assert lessons._ask_model("prompt", 60) == '{"ok": true}'
    lessons._ask_model("find videos", 60, web_search=True)
    assert calls == [("gpt-6.2-sol", "medium", True, False), ("gpt-6.2-sol", "medium", True, True)]
    assert notes.claude_fast is False, "Claude's fast mode is only for Claude models"


def test_codex_failure_explains_itself(monkeypatch):
    lessons.configure_model("gpt-6-luna", "low", False)
    monkeypatch.setattr(codex_cli, "last_failure", "GPT-6 Luna (Codex) exited with code 1: not signed in")
    assert lessons._plan_failure_message() == \
        "The lesson could not be written: GPT-6 Luna (Codex) exited with code 1: not signed in."


# --- the Codex command line ----------------------------------------------------------------------------

def test_codex_command_is_read_only_ephemeral_and_strict():
    args = codex_cli.command("C:/tmp/w", "gpt-6.1-terra", "low", True, False, "C:/tmp/w/reply.txt")
    assert args[args.index("-m") + 1] == "gpt-6.1-terra"
    joined = " ".join(args)
    for flag in ("exec", "--strict-config", "--ignore-user-config", "--ephemeral", "-s read-only",
                 'model_reasoning_effort="low"', 'web_search="disabled"', 'service_tier="fast"',
                 "-o C:/tmp/w/reply.txt"):
        assert flag in joined
    assert args[-1] == "-", "the prompt goes in on stdin, never on the command line"
    plain = " ".join(codex_cli.command("w", "gpt-6-luna", "high", False, True, "r"))
    assert "service_tier" not in plain and 'web_search="live"' in plain


def test_codex_run_reads_the_reply_file(monkeypatch):
    class FakeProcess:
        returncode = 0

        def __init__(self, args, **kwargs):
            self.args = args
            Path(args[args.index("-o") + 1]).write_text("  the reply  ", encoding="utf-8")

        def communicate(self, text, timeout=None):
            assert text.startswith("You are being used as a plain text generator") and text.endswith("PROMPT")
            return "", ""
    monkeypatch.setattr(codex_cli, "find_cli", lambda: ["node", "codex.js"])
    monkeypatch.setattr(codex_cli.subprocess, "Popen", FakeProcess)
    assert codex_cli.run("PROMPT", "gpt-6-luna", "low") == "the reply"


def test_codex_run_reports_a_failure(monkeypatch):
    class Failing:
        returncode = 1

        def __init__(self, args, **kwargs):
            pass

        def communicate(self, text, timeout=None):
            return "", "error: you are not signed in\n"
    monkeypatch.setattr(codex_cli, "find_cli", lambda: ["node", "codex.js"])
    monkeypatch.setattr(codex_cli.subprocess, "Popen", Failing)
    assert codex_cli.run("p", "gpt-6-luna") is None
    assert codex_cli.last_failure == "GPT-6 Luna (Codex) exited with code 1: error: you are not signed in"


def test_codex_refuses_a_model_it_does_not_offer():
    assert codex_cli.run("p", "gpt-5.6-sol") is None and "isn't one of the Codex models" in codex_cli.last_failure


def test_the_page_gets_the_model_list(monkeypatch):
    monkeypatch.setattr(codex_cli, "available", lambda: (True, ""))
    monkeypatch.setattr(notes, "cli_available", lambda: True)
    options = lessons.model_options()
    assert [o["id"] for o in options] == ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5",
                                          "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-terra", "gpt-6-luna"]
    assert {o["provider"] for o in options} == {"claude", "codex"} and all(o["available"] for o in options)
