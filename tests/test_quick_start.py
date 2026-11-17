"""Starting a recording with one key: the audio source and note formatting from last time are
reused without asking, and the model starts loading at the menu when a class is on. Neither
may ever record the wrong thing silently or get in the way of the other menu options."""
import json
import sys
import threading
import time

import numpy as np
import pytest

import capture
import main
import transcribe

SR = capture.audio.SAMPLE_RATE


class Args:
    def __init__(self, source=None, formatting=None):
        self.source, self.formatting, self.klass = source, formatting, None


def no_prompt(*args):
    pytest.fail(f"asked a question that was already answered: {args}")


def answers(monkeypatch, *replies):
    asked = []
    queue = iter(replies)

    def reply(prompt=""):
        asked.append(prompt)
        return next(queue)

    monkeypatch.setattr("builtins.input", reply)
    return asked


@pytest.fixture
def choices(tmp_path, monkeypatch):
    """A recorder_choices.json of this test's own, and no earlier recording to copy from."""
    path = tmp_path / "recorder_choices.json"
    monkeypatch.setattr(main, "LAST_CHOICES_PATH", path)
    monkeypatch.setattr(main.telemetry_module, "read_status", lambda: {})
    return path


# --- remembered settings ---------------------------------------------------------------------

def test_settings_from_last_time_are_used_without_asking(choices, monkeypatch):
    choices.write_text(json.dumps({"source": "system", "formatting": "local"}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", no_prompt)
    assert main.choose_source(Args()) == "system"
    assert main.choose_formatting_mode(Args()) == "local"


def test_the_first_answers_are_asked_once_then_remembered(choices, monkeypatch):
    asked = answers(monkeypatch, "2", "3")
    assert main.choose_source(Args()) == "system"
    assert main.choose_formatting_mode(Args()) == "heuristic"
    assert len(asked) == 2
    assert json.loads(choices.read_text(encoding="utf-8")) == {"source": "system", "formatting": "heuristic"}
    monkeypatch.setattr("builtins.input", no_prompt)
    assert main.choose_source(Args()) == "system"
    assert main.choose_formatting_mode(Args()) == "heuristic"


def test_flags_win_for_one_launch_and_are_not_remembered(choices, monkeypatch):
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", no_prompt)
    assert main.choose_source(Args(source="system")) == "system"
    assert main.choose_formatting_mode(Args(formatting="heuristic")) == "heuristic"
    assert json.loads(choices.read_text(encoding="utf-8")) == {"source": "mic", "formatting": "auto"}


@pytest.mark.parametrize("content", ["", "not json", "[]", "null",
                                     '{"source": "tape", "formatting": "fancy"}'])
def test_a_damaged_settings_file_just_means_the_questions_are_asked(choices, monkeypatch, content):
    choices.write_text(content, encoding="utf-8")
    assert main.load_last_choices() == {}
    answers(monkeypatch, "", "")
    assert main.choose_source(Args()) == "mic"
    assert main.choose_formatting_mode(Args()) == "auto"


def test_one_valid_setting_survives_next_to_an_invalid_one(choices):
    choices.write_text(json.dumps({"source": "system", "formatting": "fancy"}), encoding="utf-8")
    assert main.load_last_choices() == {"source": "system"}


def test_the_last_recording_counts_as_last_time_until_something_is_saved(choices, monkeypatch):
    monkeypatch.setattr(main.telemetry_module, "read_status",
                        lambda: {"source": "mic", "formatting_mode": "auto", "active": False})
    assert main.load_last_choices() == {"source": "mic", "formatting": "auto"}
    main.save_last_choices(source="system")
    assert main.load_last_choices() == {"source": "system", "formatting": "auto"}


def test_saving_never_stops_a_launch(choices, monkeypatch):
    monkeypatch.setattr(main, "LAST_CHOICES_PATH", choices.parent / "missing folder" / "x.json")
    main.save_last_choices(source="mic")  # must not raise
    assert main.load_last_choices() == {}


# --- the menu --------------------------------------------------------------------------------

@pytest.fixture
def quiet_menu(monkeypatch):
    monkeypatch.setattr(main.transcribe, "start_dependency_preparation", lambda: None)


def test_menu_says_what_enter_will_record_with(choices, quiet_menu, monkeypatch, capsys):
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    answers(monkeypatch, "")
    assert main.choose_action(Args()) == "record"
    assert "1. Record a lecture (default) - microphone, auto formatting (as last time)" in capsys.readouterr().out


def test_menu_shows_a_flag_in_place_of_the_remembered_setting(choices, quiet_menu, monkeypatch, capsys):
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    answers(monkeypatch, "")
    main.choose_action(Args(source="system"))
    assert "system audio, auto formatting" in capsys.readouterr().out


def test_menu_makes_no_promise_before_anything_is_known(choices, quiet_menu, monkeypatch, capsys):
    answers(monkeypatch, "")
    assert main.choose_action(None) == "record"
    assert "  1. Record a lecture (default)\n" in capsys.readouterr().out


def test_option_6_changes_the_settings_and_returns_to_the_menu(choices, quiet_menu, monkeypatch, capsys):
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    asked = answers(monkeypatch, "6", "2", "", "5")
    assert main.choose_action(Args()) == "record_pick"
    assert asked == ["Choose [1]: ", "Choose [1]: ", "Choose [1]: ", "Choose [1]: "]
    assert json.loads(choices.read_text(encoding="utf-8")) == {"source": "system", "formatting": "auto"}
    out = capsys.readouterr().out
    assert out.count("What would you like to do?") == 2
    assert "system audio, auto formatting (as last time)" in out


def test_option_6_offers_the_current_settings_as_the_defaults(choices, quiet_menu, monkeypatch):
    choices.write_text(json.dumps({"source": "system", "formatting": "heuristic"}), encoding="utf-8")
    asked = answers(monkeypatch, "6", "", "", "")
    assert main.choose_action(Args()) == "record"
    assert asked[1:3] == ["Choose [2]: ", "Choose [3]: "]
    assert json.loads(choices.read_text(encoding="utf-8")) == {"source": "system", "formatting": "heuristic"}


# --- not reusing a source that is plainly the wrong one ---------------------------------------

class Listening:
    """A pre-roll that has heard this: {source: True | False | None}."""

    started = 0.0  # opened long ago: nothing left to wait for

    def __init__(self, heard):
        self._heard = heard

    def has(self, source):
        return source in self._heard

    def heard(self, source):
        return self._heard[source]


def test_a_silent_remembered_source_is_not_reused_when_the_other_has_sound(choices, monkeypatch, capsys):
    choices.write_text(json.dumps({"source": "system", "formatting": "auto"}), encoding="utf-8")
    asked = answers(monkeypatch, "", "9", "1")  # Enter is not an answer here: nothing is a safe default
    assert main.choose_source(Args(), Listening({"mic": True, "system": False})) == "mic"
    assert asked == ["Choose 1 or 2: "] * 3
    assert "system audio (used last time) is silent right now, but the microphone has sound" in capsys.readouterr().out
    assert main.load_last_choices()["source"] == "mic"


def test_a_remembered_source_that_delivered_nothing_is_not_reused_when_the_other_has_sound(choices, monkeypatch):
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    answers(monkeypatch, "2")
    assert main.choose_source(Args(), Listening({"mic": None, "system": True})) == "system"


@pytest.mark.parametrize("heard", [
    {"mic": True, "system": True},     # both have sound
    {"mic": False, "system": False},   # a quiet room before the lecture: nothing says it's wrong
    {"mic": True, "system": False},    # the remembered one is the one with sound
    {"mic": False, "system": None},    # the other source has delivered nothing
    {"mic": None, "system": None},
    {"mic": False},                    # launched with only one source listening
])
def test_the_remembered_source_is_reused_unless_the_evidence_is_clear(choices, monkeypatch, heard):
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", no_prompt)
    assert main.choose_source(Args(), Listening(heard)) == "mic"


def test_a_broken_level_check_falls_back_to_the_remembered_source(choices, monkeypatch):
    class Broken(Listening):
        def heard(self, source):
            raise OSError("device went away")

    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", no_prompt)
    assert main.choose_source(Args(), Broken({"mic": 0, "system": 0})) == "mic"


class Idle:
    def start(self):
        pass

    def abandon(self):
        pass


def feed(preroll, plan, every=0.01):
    """Delivers 0.1 s blocks from a thread, as a capture would: plan = {source: level}."""
    def run():
        for _ in range(60):
            for source, level in plan.items():
                preroll._on_block(source, np.full(SR // 10, level, dtype=np.float32))
            time.sleep(every)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_enter_pressed_at_once_still_catches_the_wrong_source(choices, monkeypatch, capsys):
    """Enter before any audio has arrived: the check waits for enough to judge."""
    monkeypatch.setattr(main, "SOURCE_CHECK_READY_SECONDS", 1.5)
    choices.write_text(json.dumps({"source": "system", "formatting": "auto"}), encoding="utf-8")
    preroll = capture.Preroll(["mic", "system"], lambda source, on_block: Idle())
    assert preroll.heard("mic") is None
    feed(preroll, {"mic": 0.1, "system": 0.0})
    answers(monkeypatch, "1")
    assert main.choose_source(Args(), preroll) == "mic"
    assert "is silent right now" in capsys.readouterr().out
    preroll.discard(everything=True)


def test_the_wait_is_bounded_when_a_source_delivers_nothing(choices, monkeypatch):
    monkeypatch.setattr(main, "SOURCE_CHECK_READY_SECONDS", 0.5)
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", no_prompt)
    preroll = capture.Preroll(["mic", "system"], lambda source, on_block: Idle())
    feed(preroll, {"mic": 0.1})
    started = time.monotonic()
    assert main.choose_source(Args(), preroll) == "mic"
    assert time.monotonic() - started < 0.8
    preroll.discard(everything=True)


def test_no_wait_once_the_window_has_been_open_long_enough(choices, monkeypatch):
    monkeypatch.setattr(main, "SOURCE_CHECK_READY_SECONDS", 30)
    choices.write_text(json.dumps({"source": "mic", "formatting": "auto"}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", no_prompt)
    preroll = capture.Preroll(["mic", "system"], lambda source, on_block: Idle())
    preroll.started -= 60
    started = time.monotonic()
    assert main.choose_source(Args(), preroll) == "mic"
    assert time.monotonic() - started < 0.1
    preroll.discard(everything=True)


def test_preroll_knows_whether_a_source_has_had_any_sound():
    preroll = capture.Preroll(["mic", "system"], lambda source, on_block: Idle())
    quiet = np.zeros(SR // 2, dtype=np.float32)
    speech = np.full(SR // 2, 0.1, dtype=np.float32)

    preroll._on_block("mic", speech)
    assert preroll.heard("mic") is None, "half a second is too little to judge"
    assert preroll.heard("system") is None
    preroll._on_block("mic", quiet)
    assert preroll.heard("mic") is True
    for _ in range(4):
        preroll._on_block("system", quiet)
    assert preroll.heard("system") is False
    preroll._on_block("system", speech)
    assert preroll.heard("system") is True, "one spoken word in the last 3 s counts"
    for _ in range(6):
        preroll._on_block("mic", quiet)
    assert preroll.heard("mic") is False, "sound from more than 3 s ago no longer counts"
    preroll.discard(everything=True)
    assert preroll.heard("mic") is None


# --- the model starts loading at the menu when a class is on ------------------------------------

@pytest.fixture
def class_is_on(monkeypatch):
    started = []
    monkeypatch.setattr(main, "PRELOAD_MODEL_AT_MENU", True)
    monkeypatch.setattr(main.sched, "find_current_class",
                        lambda: {"code": "CHEM 1450", "title": "Chemistry", "status": "in_session"})
    monkeypatch.setattr(main.telemetry_module, "read_status", lambda: {})
    monkeypatch.setattr(main.transcribe, "start_model_preload", lambda: started.append(1))
    return started


def test_model_load_starts_at_the_menu_when_a_class_is_on(class_is_on, capsys):
    assert main.start_model_early_if_class_is_on() is True
    assert class_is_on == [1]
    assert "CHEM 1450 is in session, so the transcription model is loading already." in capsys.readouterr().out


def test_no_early_load_outside_class_time(class_is_on, monkeypatch, capsys):
    monkeypatch.setattr(main.sched, "find_current_class", lambda: None)
    assert main.start_model_early_if_class_is_on() is False
    assert class_is_on == [] and capsys.readouterr().out == ""


def test_no_early_load_while_another_window_is_recording(class_is_on, monkeypatch):
    monkeypatch.setattr(main.telemetry_module, "read_status",
                        lambda: {"active": True, "updated_at": time.time() - 5})
    assert main.start_model_early_if_class_is_on() is False
    assert class_is_on == []
    # A recording that crashed hours ago left "active" behind; that must not switch this off.
    monkeypatch.setattr(main.telemetry_module, "read_status",
                        lambda: {"active": True, "updated_at": time.time() - 3600})
    assert main.start_model_early_if_class_is_on() is True


def test_a_schedule_that_cannot_be_read_never_reaches_the_menu(class_is_on, monkeypatch):
    def broken():
        raise ValueError("schedule.json is not valid JSON")

    monkeypatch.setattr(main.sched, "find_current_class", broken)
    assert main.start_model_early_if_class_is_on() is False
    assert class_is_on == []


def test_early_load_can_be_switched_off(class_is_on, monkeypatch):
    monkeypatch.setattr(main, "PRELOAD_MODEL_AT_MENU", False)
    assert main.start_model_early_if_class_is_on() is False
    assert class_is_on == []


@pytest.fixture
def slow_model(monkeypatch):
    """A model whose load blocks until released, and a record of loads and unloads."""
    state = {"loads": 0, "unloads": 0, "entered": threading.Event(), "release": threading.Event(),
             "fail": False}

    def load():
        state["loads"] += 1
        state["entered"].set()
        assert state["release"].wait(3)
        if state["fail"]:
            raise RuntimeError("the transcription model could not load - MemoryError: no VRAM")
        transcribe._model = object()
        return transcribe._model

    def unload():
        state["unloads"] += 1
        transcribe._model = None

    monkeypatch.setattr(transcribe, "_model", None)
    monkeypatch.setattr(transcribe, "_preload", None)
    monkeypatch.setattr(transcribe, "get_model", load)
    monkeypatch.setattr(transcribe, "unload_model", unload)
    return state


def wait_for(condition, seconds=3.0):
    deadline = time.time() + seconds
    while time.time() < deadline and not condition():
        time.sleep(0.005)
    assert condition()


def test_early_load_runs_once_in_the_background_and_reports_its_time(slow_model):
    transcribe.start_model_preload()
    transcribe.start_model_preload()
    assert slow_model["entered"].wait(3), "the load started without anyone waiting for it"
    assert transcribe.preload_seconds() is None
    slow_model["release"].set()
    wait_for(lambda: transcribe.preload_seconds() is not None)
    assert slow_model["loads"] == 1 and slow_model["unloads"] == 0
    assert transcribe._model is not None


def test_choosing_something_else_frees_the_model_once_its_load_finishes(slow_model):
    transcribe.start_model_preload()
    assert slow_model["entered"].wait(3)
    started = time.monotonic()
    transcribe.cancel_model_preload()
    assert time.monotonic() - started < 0.5, "cancelling never waits for the load"
    assert slow_model["unloads"] == 0
    slow_model["release"].set()
    wait_for(lambda: slow_model["unloads"] == 1)
    assert transcribe._model is None


def test_choosing_something_else_after_the_load_finished_frees_it_too(slow_model):
    slow_model["release"].set()
    transcribe.start_model_preload()
    wait_for(lambda: transcribe.preload_seconds() is not None)
    wait_for(lambda: transcribe._preload._finished)
    transcribe.cancel_model_preload()
    assert slow_model["unloads"] == 1 and transcribe._model is None


def test_a_failed_early_load_is_silent_and_leaves_the_real_check_to_report_it(slow_model):
    slow_model["fail"] = True
    slow_model["release"].set()
    transcribe.start_model_preload()
    wait_for(lambda: transcribe._preload._finished)
    assert transcribe.preload_seconds() is None and transcribe._model is None
    assert slow_model["unloads"] == 0


def test_cancelling_with_no_early_load_does_nothing(slow_model):
    transcribe.cancel_model_preload()
    assert slow_model["loads"] == 0 and slow_model["unloads"] == 0


@pytest.mark.parametrize("action", ["study_guide", "flashcards", "dashboard"])
def test_other_menu_options_release_the_early_load(action, monkeypatch):
    cancelled = []
    monkeypatch.setattr(main, "start_preroll", lambda args: None)
    monkeypatch.setattr(main, "choose_action", lambda args: action)
    monkeypatch.setattr(main, "choose_notes_class", lambda args: "TEST 100")
    monkeypatch.setattr(main, "run_study_guide", lambda code: None)
    monkeypatch.setattr(main, "run_flashcards", lambda code: None)
    import dashboard
    monkeypatch.setattr(dashboard, "serve", lambda: None)
    monkeypatch.setattr(main.transcribe, "cancel_model_preload", lambda: cancelled.append(1))
    monkeypatch.setattr(sys, "argv", ["main.py"])
    main.run()
    assert cancelled == [1]


# --- the startup check ------------------------------------------------------------------------

def test_a_brief_memory_shortage_is_retried_as_the_real_loader_reports_it(monkeypatch):
    """transcribe.get_model() reports every failure as a RuntimeError caused by the original,
    so the retry has to look at the cause. (It only looked for a bare MemoryError, which the
    real loader never raises, so a shortage failed at once.)"""
    calls = []

    def load():
        calls.append(1)
        if len(calls) < 3:
            try:
                raise MemoryError("Insufficient free VRAM for Cohere plus 1 GiB headroom")
            except MemoryError as shortage:
                raise RuntimeError(f"the transcription model could not load - MemoryError: {shortage}") from shortage

    monkeypatch.setattr(main.transcribe, "get_model", load)
    monkeypatch.setattr(main.transcribe, "backend_name", lambda: "cohere")
    monkeypatch.setattr(main.transcribe, "device_info", lambda: "cuda")
    monkeypatch.setattr(main, "MODEL_LOAD_RETRY_SECONDS", 0)
    blocks, _ = main._check_transcription_model()
    assert not blocks and len(calls) == 3


def test_a_failure_that_waiting_cannot_fix_is_not_retried(monkeypatch):
    calls = []

    def load():
        calls.append(1)
        try:
            raise FileNotFoundError("model files not found")
        except FileNotFoundError as missing:
            raise RuntimeError(f"the transcription model could not load - {missing}") from missing

    monkeypatch.setattr(main.transcribe, "get_model", load)
    monkeypatch.setattr(main, "MODEL_LOAD_RETRY_SECONDS", 0)
    blocks, message = main._check_transcription_model()
    assert blocks and len(calls) == 1 and "model files not found" in message


def test_timing_log_says_when_the_model_was_loaded_at_the_menu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "STARTUP_TIMING_LOG", tmp_path / "startup_timings.jsonl")
    monkeypatch.setattr(main, "_check_transcription_model", lambda: (False, "  model ok"))
    monkeypatch.setattr(main, "_check_audio_device", lambda source, capture=None: (False, "  audio ok"))
    monkeypatch.setattr(main, "_check_disk_space", lambda: None)
    monkeypatch.setattr(main.mic_mute, "check_and_unmute", lambda: "ok")
    monkeypatch.setattr(main.transcribe, "device_info", lambda: "cuda/bfloat16/nano/bulk")
    monkeypatch.setattr(main.transcribe, "preload_seconds", lambda: 6.5)
    assert main.PreflightRunner().finish("mic", "heuristic") is True
    entry = json.loads((tmp_path / "startup_timings.jsonl").read_text(encoding="utf-8"))
    assert entry["model_load_at_menu"] == 6.5 and entry["ok"] is True
    assert "the model had started loading at the menu and took 6.5s" in capsys.readouterr().out

    monkeypatch.setattr(main.transcribe, "preload_seconds", lambda: None)
    (tmp_path / "startup_timings.jsonl").unlink()
    main.PreflightRunner().finish("mic", "heuristic")
    assert "model_load_at_menu" not in json.loads((tmp_path / "startup_timings.jsonl").read_text(encoding="utf-8"))
