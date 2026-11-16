"""Keeps every test away from the real recorder's live files.

The recorder and dashboard talk through state/status.json and state/command.json, and
TranscriptionWorker appends timings under state/. A per-test monkeypatch isn't enough on
its own: a background thread a test forgot to stop (e.g. a telemetry heartbeat) keeps
running after monkeypatch restores the real paths, and would then write fake status into
a live recording's dashboard. These paths are pointed at a temp folder once for the whole
run and deliberately never restored. Tests can still monkeypatch them further.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import performance
import telemetry


@pytest.fixture(scope="session", autouse=True)
def isolate_live_state_files(tmp_path_factory):
    state = tmp_path_factory.mktemp("isolated_state")
    telemetry.STATE_DIR = state
    telemetry.STATUS_PATH = state / "status.json"
    telemetry.COMMAND_PATH = state / "command.json"
    performance.PATH = state / "timings.jsonl"
    # Installed optional assets must not turn mocked recorder tests into GPU jobs.
    import alignment
    alignment.available = lambda: False
    import native_speech
    native_speech.available = lambda: False
    import vocabulary_retry
    vocabulary_retry.available = lambda: False
    # A recorder test that fails preflight on purpose wrote its reason into the real
    # state/startup_failures.log, next to genuine failures.
    import main
    main.STARTUP_FAILURE_LOG = state / "startup_failures.log"
    # The end of a recorded session starts a background cleanup (repair.py) and the dashboard
    # starts one on launch; from a test either would run against the real notes and state.
    import repair
    repair.STATE_DIR = state
    repair.JOB_PATH = state / "repair_job.json"
    repair.MEMORY_PATH = state / "repair_memory.json"
    repair._spawn_detached = lambda extra: None
    import archive
    archive.STATE_DIR = state
    import dashboard
    dashboard.start_auto_repair = lambda: None
    yield
