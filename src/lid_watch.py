"""Tells the recorder when the laptop lid is closed.

Windows reports the lid through a power-setting notification (GUID_LIDSWITCH_STATE_CHANGE),
which needs a window to receive it. This makes an invisible message-only window on its own
thread, so nothing else in the app has to run a Windows message loop.

Only a lid that goes from open to closed counts. The first report after registering is just
the lid's current state, so starting a recording with the lid already closed (a laptop on a
dock, say) doesn't stop it. Everything is best effort: if the machine reports no lid, or the
window can't be made, `available` stays False and the recorder works exactly as before.
"""
import ctypes
import sys
import threading
from ctypes import wintypes

WM_POWERBROADCAST = 0x0218
PBT_POWERSETTINGCHANGE = 0x8013
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
HWND_MESSAGE = -3
LID_CLOSED, LID_OPEN = 0, 1


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


# GUID_LIDSWITCH_STATE_CHANGE {BA3E0F4D-B817-4094-A2D1-D56379E6A0F3}
LID_GUID_BYTES = bytes.fromhex("4d0f3eba17b89440a2d1d56379e6a0f3")  # in-memory (little-endian) layout


def _guid() -> _GUID:
    guid = _GUID()
    ctypes.memmove(ctypes.byref(guid), LID_GUID_BYTES, 16)
    return guid


def parse_power_setting(buffer: bytes):
    """The lid state (LID_CLOSED / LID_OPEN) in a POWERBROADCAST_SETTING, or None if it's another setting."""
    if len(buffer) < 21 or buffer[:16] != LID_GUID_BYTES:
        return None
    length = int.from_bytes(buffer[16:20], "little")
    if length < 1:
        return None
    return int.from_bytes(buffer[20:20 + min(length, 4)], "little")


class LidWatcher:
    """`on_close()` runs on the watcher's thread each time the lid goes from open to closed."""

    def __init__(self, on_close, on_open=None):
        self.on_close = on_close
        self.on_open = on_open
        self.available = False     # True once the lid notification is registered
        self.state = None          # LID_OPEN / LID_CLOSED once Windows has reported it
        self._hwnd = None
        self._ready = threading.Event()
        self._thread = None
        self._proc = None          # kept referenced so the callback isn't garbage collected

    def start(self, timeout: float = 3.0) -> bool:
        if sys.platform != "win32":
            return False
        self._thread = threading.Thread(target=self._run, daemon=True, name="lid-watch")
        self._thread.start()
        self._ready.wait(timeout)
        return self.available

    def stop(self):
        hwnd = self._hwnd
        if hwnd:
            try:
                ctypes.windll.user32.PostMessageW(wintypes.HWND(hwnd), WM_CLOSE, 0, 0)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)

    def handle(self, state: int):
        """Applies one lid report. Split out so it can be driven without Windows."""
        previous, self.state = self.state, state
        if previous is None:
            return  # the first report is just where the lid already is
        try:
            if state == LID_CLOSED and previous != LID_CLOSED:
                self.on_close()
            elif state == LID_OPEN and previous == LID_CLOSED and self.on_open:
                self.on_open()
        except Exception:
            pass  # a faulty handler must never take down the watcher thread

    def _run(self):
        try:
            self._loop()
        except Exception:
            pass
        finally:
            self._ready.set()

    def _loop(self):
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        LRESULT = ctypes.c_ssize_t
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASS(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                        ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HANDLE),
                        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HANDLE),
                        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.DefWindowProcW.restype = LRESULT
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                           wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.RegisterPowerSettingNotification.argtypes = [wintypes.HANDLE, ctypes.POINTER(_GUID), wintypes.DWORD]
        user32.RegisterPowerSettingNotification.restype = wintypes.HANDLE
        user32.UnregisterPowerSettingNotification.argtypes = [wintypes.HANDLE]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.PostQuitMessage.argtypes = [ctypes.c_int]
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE  # the default int return cuts a 64-bit handle in half

        def proc(hwnd, message, wparam, lparam):
            if message == WM_POWERBROADCAST and wparam == PBT_POWERSETTINGCHANGE and lparam:
                try:
                    header = ctypes.string_at(lparam, 20)
                    length = int.from_bytes(header[16:20], "little")
                    state = parse_power_setting(ctypes.string_at(lparam, 20 + min(length, 4)))
                    if state is not None:
                        self.handle(state)
                except Exception:
                    pass
                return 1
            if message == WM_CLOSE:
                user32.DestroyWindow(hwnd)
                return 0
            if message == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        self._proc = WNDPROC(proc)
        instance = kernel32.GetModuleHandleW(None)
        window_class = WNDCLASS()
        window_class.lpfnWndProc = self._proc
        window_class.hInstance = instance
        window_class.lpszClassName = f"LectureNotesLidWatch{id(self)}"
        if not user32.RegisterClassW(ctypes.byref(window_class)):
            return
        hwnd = user32.CreateWindowExW(0, window_class.lpszClassName, "lid", 0, 0, 0, 0, 0,
                                      wintypes.HWND(HWND_MESSAGE), None, instance, None)
        if not hwnd:
            return
        guid = _guid()
        registration = user32.RegisterPowerSettingNotification(hwnd, ctypes.byref(guid), 0)  # 0 = a window handle
        if not registration:
            user32.DestroyWindow(hwnd)
            return
        self._hwnd = hwnd
        self.available = True
        self._ready.set()
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.DispatchMessageW(ctypes.byref(message))
        user32.UnregisterPowerSettingNotification(registration)
        self._hwnd = None
