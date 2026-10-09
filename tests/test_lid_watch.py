"""Closing the laptop lid: stop and save a recording, but never touch a lesson build."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lid_watch
import lessons


def _setting(state: int, guid: bytes = lid_watch.LID_GUID_BYTES) -> bytes:
    return guid + (4).to_bytes(4, "little") + state.to_bytes(4, "little")


class TestLidReports:
    def test_reads_the_lid_state_and_ignores_other_settings(self):
        assert lid_watch.parse_power_setting(_setting(0)) == lid_watch.LID_CLOSED
        assert lid_watch.parse_power_setting(_setting(1)) == lid_watch.LID_OPEN
        assert lid_watch.parse_power_setting(_setting(1, guid=bytes(16))) is None
        assert lid_watch.parse_power_setting(b"short") is None

    def test_only_an_open_to_closed_change_stops_the_recording(self):
        events = []
        watcher = lid_watch.LidWatcher(lambda: events.append("closed"), lambda: events.append("open"))
        watcher.handle(lid_watch.LID_CLOSED)   # first report: where the lid already is (a dock, say)
        assert events == []                    # so starting a recording with the lid shut doesn't stop it
        watcher.handle(lid_watch.LID_OPEN)
        assert events == ["open"]
        watcher.handle(lid_watch.LID_CLOSED)   # now a real close
        watcher.handle(lid_watch.LID_CLOSED)   # repeated reports don't fire twice
        assert events == ["open", "closed"]

    def test_a_lid_that_starts_open_then_closes_fires_once(self):
        events = []
        watcher = lid_watch.LidWatcher(lambda: events.append("closed"))
        watcher.handle(lid_watch.LID_OPEN)
        assert events == []
        watcher.handle(lid_watch.LID_CLOSED)
        assert events == ["closed"]

    def test_a_failing_handler_does_not_break_the_watcher(self):
        def boom():
            raise RuntimeError("handler failed")
        watcher = lid_watch.LidWatcher(boom)
        watcher.handle(lid_watch.LID_OPEN)
        watcher.handle(lid_watch.LID_CLOSED)   # must not raise
        assert watcher.state == lid_watch.LID_CLOSED

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows power notifications")
    def test_registers_with_windows_and_shuts_down_cleanly(self):
        watcher = lid_watch.LidWatcher(lambda: None)
        started = watcher.start()
        watcher.stop()
        assert watcher._thread is None or not watcher._thread.is_alive()
        assert started in (True, False)  # a PC with no lid switch reports unavailable, not an error


class TestSleepBlocks:
    def test_a_lesson_build_blocks_sleep_while_it_runs_and_releases_it_after(self, monkeypatch, tmp_path):
        import json
        calls = []
        monkeypatch.setattr(lessons.keep_awake, "prevent_sleep", lambda: calls.append("block") or True)
        monkeypatch.setattr(lessons.keep_awake, "allow_sleep", lambda: calls.append("release"))

        def fake_build(*args, **kwargs):
            calls.append("build")
            return "lesson-id"
        monkeypatch.setattr(lessons, "build_lesson", fake_build)
        status = tmp_path / "job.json"
        status.write_text(json.dumps({"id": "j", "status": "running", "class_code": "CHEM 1450",
                                      "start": "2026-09-09", "end": "2026-09-09", "created_at": 0}), encoding="utf-8")
        assert lessons.run_job(status) == 0
        assert calls == ["block", "build", "release"]

    def test_the_block_is_released_when_a_build_fails(self, monkeypatch, tmp_path):
        import json
        calls = []
        monkeypatch.setattr(lessons.keep_awake, "prevent_sleep", lambda: calls.append("block") or True)
        monkeypatch.setattr(lessons.keep_awake, "allow_sleep", lambda: calls.append("release"))
        monkeypatch.setattr(lessons, "build_lesson", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
        status = tmp_path / "job.json"
        status.write_text(json.dumps({"id": "j", "status": "running", "class_code": "CHEM 1450",
                                      "start": "2026-09-09", "end": "2026-09-09", "created_at": 0}), encoding="utf-8")
        assert lessons.run_job(status) == 1
        assert calls == ["block", "release"]
