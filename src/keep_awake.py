"""Prevents Windows from *system*-sleeping while a lecture is being recorded, without
forcing the display to stay on. Without this, an idle timeout can suspend the whole
process mid-recording - which silently kills transcription until the machine wakes back
up. Uses the standard Win32 SetThreadExecutionState API (the same mechanism video
players and download managers use to stay awake during long-running tasks).

Deliberately does NOT set ES_DISPLAY_REQUIRED: the app doesn't need the screen on to
keep recording/transcribing in the background, and a laptop display is one of the
biggest power draws there is - forcing it on for a 50+ minute lecture would waste real
battery for no benefit. Let the screen turn off on its own; only the system needs to
stay awake.
"""
import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002  # intentionally not used - see module docstring


def prevent_sleep():
    """Blocks system sleep (screen may still turn off normally) until allow_sleep() is
    called. No-op (returns False) on non-Windows platforms or if the call fails."""
    if sys.platform != "win32":
        return False
    try:
        result = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        return result != 0
    except Exception:
        return False


def allow_sleep():
    """Releases the sleep block, restoring normal power-management behavior."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


class KeepAwake:
    """Context manager: `with KeepAwake(): ...` blocks system sleep for the duration
    (screen can still turn off normally)."""

    def __enter__(self):
        self.active = prevent_sleep()
        return self

    def __exit__(self, *exc_info):
        allow_sleep()
