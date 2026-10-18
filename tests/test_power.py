"""Optional background work stands down on battery."""
import ctypes
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import power


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power status")
@pytest.mark.parametrize("line_status, expected", [(0, True), (1, False), (255, False)])
def test_on_battery_follows_the_ac_line_status(monkeypatch, line_status, expected):
    def fake(pointer):
        ctypes.cast(pointer, ctypes.POINTER(power._PowerStatus)).contents.ACLineStatus = line_status
        return 1
    monkeypatch.setattr(ctypes.windll.kernel32, "GetSystemPowerStatus", fake, raising=False)
    assert power.on_battery() is expected


def test_a_failed_query_is_not_treated_as_battery(monkeypatch):
    if sys.platform == "win32":
        monkeypatch.setattr(ctypes.windll.kernel32, "GetSystemPowerStatus", lambda p: 0, raising=False)
    assert power.on_battery() is False


def test_captions_option_defaults_to_plugged_in_only():
    source = (Path(__file__).resolve().parent.parent / "src/main.py").read_text(encoding="utf-8")
    assert "choices=['auto', 'on', 'off']" in source
    assert "args.captions == 'auto' and not power.on_battery()" in source
