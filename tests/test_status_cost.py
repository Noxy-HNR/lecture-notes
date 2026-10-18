"""The Diagnostics page polls /api/status every second, so a poll must stay cheap."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import dashboard
import gpu_formatter


def test_slow_status_is_reused_within_the_window_and_refreshed_after(monkeypatch):
    calls = []
    monkeypatch.setattr(dashboard, "service_status", lambda: calls.append("s") or {"claude_cli": "available"})
    monkeypatch.setattr(dashboard.recorder_launcher, "recorder_state", lambda: calls.append("r") or {"running": False})
    dashboard.forget_slow_status()
    first = dashboard.slow_status(now=100.0)
    assert dashboard.slow_status(now=100.0 + dashboard.SLOW_STATUS_SECONDS - 0.1) == first
    assert calls == ["s", "r"]                       # probed once
    dashboard.slow_status(now=100.0 + dashboard.SLOW_STATUS_SECONDS + 0.1)
    assert calls == ["s", "r", "s", "r"]             # refreshed after the window
    dashboard.forget_slow_status()
    dashboard.slow_status(now=100.0 + dashboard.SLOW_STATUS_SECONDS + 0.2)
    assert len(calls) == 6                           # starting the recorder drops the cached answer


def test_stopped_local_model_is_reported_without_a_connection_attempt(monkeypatch):
    # A TCP connect to a closed port can block ~2 s on Windows; the listener table answers first.
    monkeypatch.setattr(gpu_formatter, "available", lambda: True)
    monkeypatch.setattr(gpu_formatter, "_port_open", lambda: False)
    monkeypatch.setattr(gpu_formatter, "_server_healthy",
                        lambda: (_ for _ in ()).throw(AssertionError("must not connect to a closed port")))
    assert dashboard.service_status()["gpu_model"] == "installed (stopped)"


def test_running_local_model_still_gets_its_health_check(monkeypatch):
    monkeypatch.setattr(gpu_formatter, "available", lambda: True)
    monkeypatch.setattr(gpu_formatter, "_port_open", lambda: True)
    monkeypatch.setattr(gpu_formatter, "_server_healthy", lambda: True)
    assert dashboard.service_status()["gpu_model"] == "running"


def test_pages_do_not_poll_while_their_tab_is_hidden():
    diagnostics = (ROOT / "dashboard/diagnostics.html").read_text(encoding="utf-8")
    assert not re.search(r"setInterval\((poll|loadJev|loadHealth),", diagnostics)
    assert "pollWhileVisible(poll, 1000)" in diagnostics and "document.hidden" in diagnostics
    lessons = (ROOT / "dashboard/lessons.html").read_text(encoding="utf-8")
    assert "setInterval(tick," not in lessons and "document.hidden" in lessons
