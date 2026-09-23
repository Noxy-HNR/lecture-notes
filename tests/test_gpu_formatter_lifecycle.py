"""Lifecycle guarantees for Lecture Notes' privately owned llama-server process."""
import subprocess
import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gpu_formatter


class FakeProcess:
    def __init__(self, running=True, stubborn=False):
        self.running = running
        self.stubborn = stubborn
        self.terminated = 0
        self.killed = 0
        self.waits = []

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated += 1
        if not self.stubborn:
            self.running = False

    def kill(self):
        self.killed += 1
        self.running = False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.running:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return 0


def _install_owned(monkeypatch, process, job=123):
    stopped = type("StopEvent", (), {"set": lambda self: setattr(self, "was_set", True)})()
    stopped.was_set = False
    monkeypatch.setattr(gpu_formatter, "_server_process", process)
    monkeypatch.setattr(gpu_formatter, "_server_job", job)
    monkeypatch.setattr(gpu_formatter, "_watchdog_stop_event", stopped)
    closed = []
    monkeypatch.setattr(gpu_formatter, "_close_windows_job", closed.append)
    return stopped, closed


def test_stop_server_terminates_owned_process_and_closes_job(monkeypatch):
    process = FakeProcess()
    stopped, closed = _install_owned(monkeypatch, process)

    gpu_formatter.stop_server()

    assert process.terminated == 1
    assert process.killed == 0
    assert process.waits == [5]
    assert stopped.was_set
    assert closed == [123]
    assert gpu_formatter._server_process is None
    assert gpu_formatter._server_job is None


def test_stop_server_kills_owned_process_that_ignores_terminate(monkeypatch):
    process = FakeProcess(stubborn=True)
    _, closed = _install_owned(monkeypatch, process)

    gpu_formatter.stop_server()

    assert process.terminated == 1
    assert process.killed == 1
    assert process.waits == [5, 5]
    assert closed == [123]


def test_stop_server_never_targets_healthy_server_not_started_here(monkeypatch):
    """A healthy process on the port is borrowed; without an owned Popen it is untouched."""
    stopped, closed = _install_owned(monkeypatch, None, job=None)

    gpu_formatter.stop_server()

    assert stopped.was_set
    assert closed == [None]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_windows_kill_job_reaps_owned_child_when_handle_closes():
    # Use a direct executable rather than the venv's python.exe launcher: that launcher
    # can hand off to another interpreter process and exit successfully before the job
    # is closed, which would test the launcher rather than Job Object cleanup.
    process = subprocess.Popen(
        [str(Path(os.environ["WINDIR"]) / "System32" / "ping.exe"), "-t", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    job = None
    try:
        job = gpu_formatter._attach_windows_kill_job(process)
        assert job is not None, "the formatter child could not be assigned to its kill-on-close job"
        gpu_formatter._close_windows_job(job)
        job = None
        process.wait(timeout=5)
        assert process.poll() is not None
    finally:
        if job is not None:
            gpu_formatter._close_windows_job(job)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
