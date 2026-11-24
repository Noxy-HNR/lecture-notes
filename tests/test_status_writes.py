"""Status files the dashboard polls are swapped in atomically. Windows refuses the swap for a
moment while another process has the file open; that once stopped a whole lesson build
("[WinError 5] Access is denied: ...json.tmp -> ...json")."""
import json
import os

import pytest

import lessons
import repair
import storage


def flaky_replace(monkeypatch, refusals):
    """os.replace that is refused `refusals` times, then works."""
    real = os.replace
    calls = []

    def replace(src, dst):
        calls.append(1)
        if len(calls) <= refusals:
            raise PermissionError(5, "Access is denied", str(src))
        return real(src, dst)
    monkeypatch.setattr(storage.os, "replace", replace)
    monkeypatch.setattr(storage, "REPLACE_RETRY_SECONDS", 0)
    return calls


def test_a_briefly_refused_swap_is_retried(tmp_path, monkeypatch):
    calls = flaky_replace(monkeypatch, refusals=3)
    lessons._write_json(tmp_path / "job.json", {"status": "running"})
    assert json.loads((tmp_path / "job.json").read_text(encoding="utf-8")) == {"status": "running"}
    assert len(calls) == 4


def test_a_progress_update_never_stops_a_build(tmp_path, monkeypatch, capsys):
    flaky_replace(monkeypatch, refusals=10 ** 6)   # refused every time
    job = {"status_path": str(tmp_path / "job.json"), "status": "running"}
    lessons._update(job, narration={"done": 3, "total": 25})   # must not raise
    assert job["narration"] == {"done": 3, "total": 25}
    assert "Couldn't save build progress" in capsys.readouterr().err


def test_the_cleanup_job_file_is_retried_too(tmp_path, monkeypatch):
    calls = flaky_replace(monkeypatch, refusals=2)
    repair._write_json(tmp_path / "repair_job.json", {"status": "running"})
    assert json.loads((tmp_path / "repair_job.json").read_text(encoding="utf-8")) == {"status": "running"}
    assert len(calls) == 3


def test_a_swap_refused_for_too_long_still_fails_loudly_outside_progress(tmp_path, monkeypatch):
    flaky_replace(monkeypatch, refusals=10 ** 6)
    with pytest.raises(PermissionError):
        storage.replace_with_retry(tmp_path / "missing.tmp", tmp_path / "x.json")
