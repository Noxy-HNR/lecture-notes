"""Whether the PC is running on battery, so optional background work can stand down.

Windows only (GetSystemPowerStatus); anywhere else, or if the call fails, the answer is
"not on battery" so nothing optional is switched off by mistake."""
import ctypes
import sys


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def on_battery() -> bool:
    if sys.platform != "win32":
        return False
    try:
        status = _PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return False
        return status.ACLineStatus == 0      # 1 = plugged in, 255 = unknown
    except Exception:
        return False
