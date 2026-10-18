"""Offline lecture formatting through the shared dashboard-managed llama.cpp server.

Only the saved lean Bonsai 27B profile is eligible. Reuse requires its exact model
path and an idle slot; loading is allowed only when the manager is idle. This client
never launches a model process and never changes saved settings. Shutdown leaves the
shared server running. Critical RAM asks the manager to stop only the exact original
process after identity checks. A missing/busy manager or insufficient RAM returns None.
The Windows job helpers remain solely for the separate owned caption worker.
"""
import ctypes
import http.client
import json
import math
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_CONFIG_PATH = Path("C:/AI/Config/models.json")
MODELS_DIR = Path("C:/AI/models")
MODEL_FILENAME = "Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
PORT = 10000
BASE_URL = f"http://127.0.0.1:{PORT}"
MANAGER_URL = "http://127.0.0.1:9090"
MANAGER_SCRIPT = Path("C:/AI/dashboard-server.ps1")
MANAGER_SHELL = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
MANAGER_STATE_DIR = PROJECT_ROOT / "state" / "shared_manager"
MANAGER_STARTUP_TIMEOUT_SECONDS = 15
STARTUP_TIMEOUT_SECONDS = 90
CONTROL_TIMEOUT_SECONDS = 5
LOAD_TIMEOUT_SECONDS = 45
CONTEXT_SIZE = 32768
REQUEST_TIMEOUT_SECONDS = 180
IDLE_TIMEOUT_SECONDS = 600  # Public compatibility only; the manager owns server idle policy.
_WATCHDOG_POLL_SECONDS = 1
_RAM_CRITICAL_BYTES = int(0.8 * 2**30)
_RAM_CRITICAL_SECONDS = 3
_server_lock = threading.RLock()
_last_used = 0.0
_last_error = None


def _available_physical_bytes():
    """Windows available physical RAM, never the misleading free-memory counter."""
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in (
                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    if os.name != "nt":
        raise OSError("Windows physical-memory admission is required")
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys


def _process_identity(pid):
    """PID plus kernel creation time, so emergency cleanup cannot target a reused PID."""
    if os.name != "nt":
        return None
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        creation, end, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(end),
                                      ctypes.byref(kernel_time), ctypes.byref(user_time)):
            return None
        return pid, (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def _process_details(pid):
    """Read-only executable and immutable args of the identified llama child."""
    import psutil
    process = psutil.Process(pid)
    return process.exe(), process.cmdline()


def _verify_process_details(identity):
    """OS-only proof also works when a loading/starved model cannot answer HTTP."""
    if identity is None or _llama_process_ids() != [identity[0]]:
        return False
    if _process_identity(identity[0]) != identity:
        return False
    binary, arguments = _process_details(identity[0])
    profile, _ = _saved_profile()
    expected = Path(profile.get("serverBinary") or "C:/AI/Tools/llama-native/bin/llama-server.exe")
    if not expected.is_absolute():
        expected = Path("C:/AI") / expected
    def normal(path):
        return str(Path(path).resolve()).replace("\\", "/").casefold()
    if normal(binary) != normal(expected):
        return False
    models = []
    for index, arg in enumerate(arguments):
        if arg in ("-m", "--model"):
            models.append(arguments[index + 1] if index + 1 < len(arguments) else None)
        elif isinstance(arg, str) and arg.startswith("--model="):
            models.append(arg.split("=", 1)[1])
    if len(models) != 1 or not isinstance(models[0], str) or normal(models[0]) != normal(MODEL_PATH):
        return False
    for flag, value in (("--load-mode", "none"), ("--cache-ram", "0")):
        values = []
        for index, arg in enumerate(arguments):
            if arg == flag:
                values.append(arguments[index + 1] if index + 1 < len(arguments) else None)
            elif isinstance(arg, str) and arg.startswith(flag + "="):
                values.append(arg.split("=", 1)[1])
        if values != [value]:
            return False
    return _process_identity(identity[0]) == identity


def _verified_process(identity):
    try:
        return _verify_process_details(identity)
    except Exception:
        return False


def _emergency_stop(identity):
    """Exact original model/process only, through the manager PID guard."""
    try:
        if not _verified_process(identity):
            return False
        # Do not ask an unresponsive model for /props during an emergency. Kernel
        # creation time plus its executable/model CLI already prove its identity.
        result = _request_json(MANAGER_URL, "/api/stop", {
            "service": "llama-server", "expectedPid": identity[0]}, timeout=CONTROL_TIMEOUT_SECONDS)
        return result.get("success") is True
    except Exception:
        return False


def _llama_process_ids():
    """Read-only process inventory; None means admission cannot be established."""
    if os.name != "nt":
        return None
    from ctypes import wintypes
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return None
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel.Process32FirstW(snapshot, ctypes.byref(entry)):
            return None
        found = []
        while True:
            if entry.szExeFile.lower() == "llama-server.exe":
                found.append(entry.th32ProcessID)
            if not kernel.Process32NextW(snapshot, ctypes.byref(entry)):
                return found if ctypes.get_last_error() == 18 else None
    finally:
        kernel.CloseHandle(snapshot)


def _port_open():
    return _listening_port(PORT)


def _listening_port(port):
    """Windows can delay closed-port refusal beyond a short TCP timeout.

    Read the OS listener table instead; an inventory error still fails closed.
    """
    try:
        import psutil
        return any(c.status == psutil.CONN_LISTEN and c.laddr.port == port and
                   c.laddr.ip in ('127.0.0.1','0.0.0.0','::','::1')
                   for c in psutil.net_connections(kind='tcp'))
    except Exception:
        return True


def _manager_port_open():
    return _listening_port(9090)


def _manager_preflight(timeout=2):
    result = _request_json(MANAGER_URL, "/api/models/preflight", {"filename": MODEL_FILENAME}, timeout=timeout)
    if not isinstance(result, dict) or result.get("filename") != MODEL_FILENAME:
        raise ValueError("Management endpoint did not identify the requested saved model")
    if type(result.get("success")) is not bool or not isinstance(result.get("admission"), dict):
        raise ValueError("Management preflight response was not recognized")
    if type(result["admission"].get("allowed")) is not bool:
        raise ValueError("Management preflight admission was not recognized")
    return result


def _management_paths_verified():
    """Only these fixed local files may be launched; resolve/reparse changes fail closed."""
    if os.name != "nt":
        return False
    for path in (MANAGER_SCRIPT, MANAGER_SHELL):
        if not path.is_file():
            return False
        expected = str(path.absolute()).replace("\\", "/").casefold()
        actual = str(path.resolve(strict=True)).replace("\\", "/").casefold()
        if actual != expected:
            return False
        for part in (path, *path.parents):
            if getattr(part.lstat(), "st_file_attributes", 0) & 0x400 or part.is_symlink():
                return False
    return True


def _acquire_manager_bootstrap():
    """Serialize recorder processes while the shared management listener starts."""
    if os.name != "nt":
        raise OSError("Windows manager bootstrap is required")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, "Local\\LectureNotesSharedManagerBootstrap")
    if not handle:
        raise OSError("Could not acquire shared manager startup guard")
    status = kernel.WaitForSingleObject(handle, 0)
    if status in (0, 0x80):
        return handle, True
    kernel.CloseHandle(handle)
    if status == 258:
        return None, False
    raise OSError("Shared manager startup guard could not be verified")


def _release_manager_bootstrap(handle):
    if handle is not None:
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def _recorded_manager_running():
    """Avoid another bootstrap after a slow known controller outlives our deadline."""
    records = sorted(MANAGER_STATE_DIR.glob("manager-*.json"), key=lambda path:path.name, reverse=True)[:16]
    expected_args = ["-noprofile", "-noninteractive", "-executionpolicy", "bypass", "-file",
                     str(MANAGER_SCRIPT).replace("\\", "/").casefold()]
    for path in records:
        try:
            with path.open("rb") as source:
                raw = source.read(16385)
            if len(raw) > 16384:
                continue
            record = json.loads(raw)
            pid, creation = record.get("pid"), record.get("creation_time")
            if type(pid) is not int or pid <= 0 or type(creation) is not int:
                continue
            identity = (pid, creation)
            if _process_identity(pid) != identity:
                continue
            binary, arguments = _process_details(pid)
            actual_binary = str(Path(binary).resolve()).replace("\\", "/").casefold()
            expected_binary = str(MANAGER_SHELL.resolve()).replace("\\", "/").casefold()
            actual_args = [str(arg).replace("\\", "/").casefold() for arg in arguments[1:]]
            if actual_binary == expected_binary and actual_args == expected_args and _process_identity(pid) == identity:
                return True
        except Exception:
            continue
    return False


def _ensure_manager(minimum):
    """Bootstrap only the known management script; never launch a model directly."""
    global _last_error
    deadline = time.monotonic() + MANAGER_STARTUP_TIMEOUT_SECONDS
    try:
        return _manager_preflight(timeout=min(2, MANAGER_STARTUP_TIMEOUT_SECONDS))
    except (OSError, ValueError, TypeError, http.client.HTTPException):
        if _manager_port_open():
            _last_error = "Shared manager unavailable: port9090 belongs to an unverified or unresponsive endpoint"
            return None
    if _port_open() or _llama_process_ids() != []:
        _last_error = "Shared manager unavailable: inference is already occupied; no controller was started"
        return None
    if _available_physical_bytes() < minimum:
        _last_error = "Shared manager unavailable: insufficient available physical RAM for safe fallback"
        return None
    handle = None
    try:
        handle, owner = _acquire_manager_bootstrap()
        # Another recorder may finish startup while we acquire the named guard.
        try:
            return _manager_preflight(timeout=min(1, max(.01, deadline-time.monotonic())))
        except (OSError, ValueError, TypeError, http.client.HTTPException):
            pass
        process = None
        if owner and not _recorded_manager_running():
            if _manager_port_open() or _port_open() or _llama_process_ids() != []:
                _last_error = "Shared manager unavailable: a listener/process appeared; no controller was started"
                return None
            if _available_physical_bytes() < minimum or not _management_paths_verified():
                _last_error = "Shared manager unavailable: management path or RAM safety verification failed"
                return None
            if time.monotonic() >= deadline:
                _last_error = "Shared manager startup deadline elapsed before launch; no controller or model was started"
                return None
            MANAGER_STATE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = str(time.time_ns())
            stdout = MANAGER_STATE_DIR / ("manager-" + stamp + ".stdout.log")
            stderr = MANAGER_STATE_DIR / ("manager-" + stamp + ".stderr.log")
            with stdout.open("xb") as out, stderr.open("xb") as err:
                process = subprocess.Popen([str(MANAGER_SHELL), "-NoProfile", "-NonInteractive",
                    "-ExecutionPolicy", "Bypass", "-File", str(MANAGER_SCRIPT)],
                    cwd=str(MANAGER_SCRIPT.parent), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW)
            identity = _process_identity(process.pid)
            record = {"pid": process.pid, "creation_time": identity[1] if identity else None,
                      "script": str(MANAGER_SCRIPT), "executable": str(MANAGER_SHELL),
                      "started_unix": time.time(), "stdout": str(stdout), "stderr": str(stderr),
                      "lifecycle": "shared controller; ordinary recorder shutdown leaves it running"}
            with (MANAGER_STATE_DIR / ("manager-" + stamp + ".json")).open("x", encoding="utf-8") as output:
                json.dump(record, output, indent=2)
            if identity is None:
                _last_error = "Shared manager started but creation identity was unavailable; it was left untouched"
                return None
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                _last_error = "Shared manager unavailable: owned controller exited before readiness; see local logs"
                return None
            if _available_physical_bytes() < minimum:
                _last_error = "Shared manager unavailable: physical RAM fell below safe fallback admission"
                return None
            try:
                return _manager_preflight(timeout=min(1, max(.01, deadline-time.monotonic())))
            except (OSError, ValueError, TypeError, http.client.HTTPException):
                time.sleep(min(.25, max(0, deadline-time.monotonic())))
        _last_error = "Shared manager startup exceeded15seconds; no model load was requested; controller left untouched"
        return None
    except (OSError, ValueError, AttributeError, TypeError, http.client.HTTPException) as error:
        _last_error = "Shared manager unavailable: bootstrap failed (" + type(error).__name__ + "); no model load requested"
        return None
    finally:
        _release_manager_bootstrap(handle)


def _request_json(base, path, payload=None, timeout=CONTROL_TIMEOUT_SECONDS, watch_ram=False,
                  process_identity=None):
    """Bound total HTTP lifetime, including a trickling body, by closing its socket."""
    port = 10000 if base == BASE_URL else 9090
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    finished = threading.Event()
    aborted = threading.Event()
    transport = []

    def abort():
        aborted.set()
        sock = transport[0] if transport else connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    def monitor():
        deadline = time.monotonic() + timeout
        low_since = None
        while not finished.wait(min(_WATCHDOG_POLL_SECONDS, max(.01, deadline - time.monotonic()))):
            now = time.monotonic()
            if now >= deadline:
                abort()
                return
            if watch_ram:
                try:
                    low = _available_physical_bytes() < _RAM_CRITICAL_BYTES
                except OSError:
                    low = True
                low_since = (low_since if low_since is not None else now) if low else None
                if low_since is not None and now - low_since >= _RAM_CRITICAL_SECONDS:
                    abort()
                    _emergency_stop(process_identity)
                    return

    watchdog = threading.Thread(target=monitor, daemon=True, name="lecture-llm-request-guard")
    watchdog.start()
    try:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        connection.request("GET" if payload is None else "POST", path, body,
                           {"Content-Type": "application/json"})
        if connection.sock is not None:
            transport.append(connection.sock)
        response = connection.getresponse()
        data = response.read(16 * 1024 * 1024 + 1)
        if aborted.is_set():
            raise TimeoutError("Shared model request exceeded its deadline or RAM limit")
        if response.status != 200 or len(data) > 16 * 1024 * 1024:
            raise OSError("Shared model HTTP request was refused")
        return json.loads(data)
    finally:
        finished.set()
        connection.close()
        watchdog.join(timeout=8)


def _saved_profile():
    config = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    profile = config.get("models", {}).get(MODEL_FILENAME)
    if not isinstance(profile, dict) or not MODEL_PATH.is_file():
        raise ValueError("Saved lean Bonsai profile/model is missing")
    args = shlex.split(profile.get("extraArgs", ""), posix=False)
    for flag, expected in (("--load-mode", "none"), ("--cache-ram", "0")):
        positions = [i for i, arg in enumerate(args) if arg == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(args) or args[positions[0] + 1] != expected:
            raise ValueError("Shared profile is missing required RAM safety flags")
    floor = profile.get("minimumAvailableRamGiB", 3.3)
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor < 3.3:
        raise ValueError("Invalid saved RAM minimum")
    return profile, int(floor * 2**30)


def available() -> bool:
    """Whether the approved shared profile is installed; does not load it."""
    try:
        _saved_profile()
        return True
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def _selected_server_idle() -> bool:
    props = _request_json(BASE_URL, "/props")
    model_path = str(props.get("model_path", "")).replace("\\", "/")
    if model_path.casefold() != str(MODEL_PATH).replace("\\", "/").casefold():
        return False
    if int(props.get("default_generation_settings", {}).get("n_ctx", 0)) < CONTEXT_SIZE:
        return False
    if _request_json(BASE_URL, "/health").get("status") != "ok":
        return False
    slots = _request_json(BASE_URL, "/slots")
    return isinstance(slots, list) and bool(slots) and all(
        isinstance(slot, dict) and slot.get("is_processing") is False for slot in slots)


def _server_healthy() -> bool:
    try:
        return _selected_server_idle()
    except (OSError, ValueError, AttributeError, TypeError, http.client.HTTPException):
        return False


def _touch():
    global _last_used
    _last_used = time.time()


def warm_up() -> bool:
    """Reuse approved idle server, or ask an idle dashboard to load its saved profile."""
    return _ensure_server_running()


def _ensure_server_running() -> bool:
    global _last_error
    with _server_lock:
        try:
            _last_error = None
            _, minimum = _saved_profile()
            if _port_open():
                if _available_physical_bytes() < _RAM_CRITICAL_BYTES or not _selected_server_idle():
                    _last_error = "Local fallback deferred: selected shared model is busy, unavailable, or critically short of RAM"
                    return False
                pids = _llama_process_ids()
                identity = _process_identity(pids[0]) if isinstance(pids, list) and len(pids) == 1 else None
                if not _verified_process(identity):
                    _last_error = "Local fallback deferred: resident model process identity/safety settings could not be verified"
                    return False
                _touch()
                return True
            if _llama_process_ids() != []:
                _last_error = "Local fallback deferred: an existing or unidentified model process must exit first"
                return False
            if _available_physical_bytes() < minimum:
                _last_error = "Local fallback deferred: saved model requires more available physical RAM"
                return False
            preflight = _ensure_manager(minimum)
            if preflight is None:
                return False
            if preflight.get("success") is not True or preflight.get("admission", {}).get("allowed") is not True:
                _last_error = "Local fallback deferred: shared manager preflight refused model admission"
                return False
            required = float(preflight["admission"]["requiredGiB"]) * 2**30
            if not math.isfinite(required) or _available_physical_bytes() < max(minimum, required):
                return False
            if _port_open() or _llama_process_ids() != []:
                return False
            result = _request_json(MANAGER_URL, "/api/models/load", {
                "filename": MODEL_FILENAME, "persist": False, "onlyIfIdle": True,
                "settings": {"contextSize": CONTEXT_SIZE}}, timeout=LOAD_TIMEOUT_SECONDS)
            if result.get("success") is not True:
                return False
            loaded_pid = result.get("pid")
            if type(loaded_pid) is not int or loaded_pid <= 0:
                _last_error = "Manager load returned no exact child PID; no cleanup or replacement was attempted"
                return False
            identity = _process_identity(loaded_pid)
            if identity is None:
                _last_error = "Manager child creation time could not be verified; no replacement was attempted"
                return False
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            done = threading.Event()
            aborted = threading.Event()
            cleanup_lock = threading.Lock()
            def cleanup_startup(cause):
                global _last_error
                with cleanup_lock:
                    if aborted.is_set():
                        return
                    aborted.set()
                    stopped = _emergency_stop(identity)
                    _last_error = cause + ("; exact manager child stopped" if stopped else
                                          "; exact child cleanup could not be verified; wait before retrying")
            def startup_guard():
                low_since = None
                while not done.wait(min(.25, _WATCHDOG_POLL_SECONDS)):
                    now = time.monotonic()
                    try:
                        low = _available_physical_bytes() < _RAM_CRITICAL_BYTES
                    except OSError:
                        low = True
                    low_since = (now if low_since is None else low_since) if low else None
                    timed_out = now >= deadline
                    if timed_out or low_since is not None and now - low_since >= _RAM_CRITICAL_SECONDS:
                        cause = "Startup readiness deadline exceeded" if timed_out else "Startup RAM remained critically low"
                        cleanup_startup(cause)
                        return
            guard = threading.Thread(target=startup_guard, daemon=True, name="lecture-llm-startup-guard")
            guard.start()
            try:
                while not aborted.is_set() and time.monotonic() < deadline:
                    if _server_healthy():
                        if not aborted.is_set() and _verified_process(identity):
                            _touch()
                            return True
                        return False
                    time.sleep(.25)
                if not aborted.is_set():
                    cleanup_startup("Startup readiness deadline exceeded")
                return False
            finally:
                done.set()
                guard.join(timeout=8)
        except (OSError, ValueError, AttributeError, TypeError, KeyError, http.client.HTTPException) as error:
            if _last_error is None:
                _last_error='Local fallback admission failed ('+type(error).__name__+')'
            return False


def stop_server():
    """Compatibility hook: release local bookkeeping; never stop a shared server."""
    global _last_used
    # Inference holds _server_lock. Shutdown must not wait up to180seconds merely
    # to reset a timestamp, and there is no process ownership to synchronize now.
    _last_used = 0.0


def _chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    # Serializes this recorder's calls. The server/manager remains responsible for
    # cross-client scheduling; the pre-request slot check is deliberately fail closed.
    with _server_lock:
        if not _ensure_server_running():
            return None
        _touch()
        payload = {
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens, "temperature": 0.3, "repeat_penalty": 1.1,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            if not _selected_server_idle():
                return None
            pids = _llama_process_ids()
            if not isinstance(pids, list) or len(pids) != 1:
                return None
            identity = _process_identity(pids[0])
            if not _verified_process(identity):
                return None
            result = _request_json(BASE_URL, "/v1/chat/completions", payload,
                                   timeout=REQUEST_TIMEOUT_SECONDS, watch_ram=True,
                                   process_identity=identity)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                return None  # Never save a truncated transcript as successful notes.
            text = choice["message"]["content"].strip()
            return text or None
        except (OSError, ValueError, AttributeError, TypeError, KeyError, IndexError, http.client.HTTPException):
            return None


def _attach_windows_kill_job(process):
    """Tie an owned server to this Python process at the Windows kernel level.

    ``atexit`` is not guaranteed to run when somebody closes the recorder's console
    window.  A Job Object with KILL_ON_JOB_CLOSE is: Windows closes our non-inherited
    job handle when this process disappears and then terminates the llama-server child.
    Only the exact ``Popen`` child created below is assigned, so a separately launched
    server (including the user's server on port 10000) can never be affected.

    Returns the job handle, or ``None`` when unavailable.  Explicit shutdown and the
    atexit hook remain in place as the portable/graceful cleanup paths.
    """
    if os.name != "nt":
        return None

    from ctypes import wintypes

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    assigned = configured and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle))
    if not assigned:
        kernel32.CloseHandle(job)
        return None
    return job


def _close_windows_job(job):
    if job is not None and os.name == "nt":
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)


_FORMAT_SYSTEM_PROMPT = """Format only the facts explicitly stated in the supplied lecture transcript as markdown notes.

Rules:
- Preserve every stated fact, negation, limitation, uncertainty, number, precision and unit. Preserve who said a claim; do not drop instructor or student attribution.
- Keep the source's original sentences and wording. Change only punctuation, obvious mechanical transcription errors and Markdown formatting. Do not replace source sentences with explanatory headings, abstractions or paraphrases.
- Use full bullet points and bold key terms. Any descriptive label must reuse wording from the source. Do not create a separate key-terms list.
- Do not add definitions, classifications, mechanisms, causes, outcomes, materials, experimental context or textbook knowledge that the source did not state. A familiar scientific fact is still an addition.
- Do not infer a positive claim from a negation, resolve an uncertain value, or turn an unmeasured outcome into an observed result.
- Correct only obvious mechanical transcription errors. Preserve uncertain wording and any existing verification markers rather than guessing.
- Keep explicit student/instructor exchanges separate under exactly "### Q&A", using standalone "**Q (Student):**" and "**A (Instructor):**" lines without bullet markers. Preserve their actual question and answer, including a refusal to conclude anything.
- Do not add a title or date heading; the caller supplies the existing date. Respond only with notes, without commentary.

Example of the style to match:
Transcript: "the mitocondrion is the powerhouse of the cell. it produces energy through a process called cellular respiration."
Notes:
- The **mitochondrion** is the powerhouse of the cell.
- It produces energy through a process called **cellular respiration**."""


def _scaled_max_tokens(transcript: str, floor: int = 400, ceiling: int = 8000) -> int:
    """Notes output needs to scale with input length - a fixed small cap (this used
    to be a flat 400) truncates output for anything longer than a short chunk, most
    obviously a whole-session transcript. The prompt tells the model to keep the
    source's own sentences, so faithful notes run close to the source length, not
    half of it: measured on real lecture chunks (1,100-2,600 words), the saved notes
    came out at 0.9-1.0x the estimated input tokens, and a half-length cap cut every
    one of them off, which _chat then discards. So allow 1.25x the estimated input
    (room for bullet markup) and never more than fits beside the prompt in the
    server's context window."""
    estimated_input_tokens = len(transcript.split()) * 1.3
    room = CONTEXT_SIZE - estimated_input_tokens - 1500  # system prompt, template and safety margin
    return int(max(256, min(ceiling, room, max(floor, estimated_input_tokens * 1.25))))


def format_transcript(class_title: str, session_date: str, transcript: str) -> str | None:
    """Returns clean markdown notes for this transcript, or None if the local LLM
    server isn't available/fails - caller falls back to the heuristic formatter."""
    user_prompt = f"Class: {class_title}\n\nTranscript:\n{transcript}"
    text = _chat(_FORMAT_SYSTEM_PROMPT, user_prompt, max_tokens=_scaled_max_tokens(transcript))
    if not text:
        return None
    text=text.strip()
    heading=f"## {session_date}"
    if text.splitlines()[0]==heading:
        text='\n'.join(text.splitlines()[1:]).strip()
    # Unexpected top-level headings can invent a date/title or create a second
    # session boundary. Refuse those instead of silently saving a malformed session.
    if not text or re.search(r'^ {0,3}#{1,2}(?:[ \t]+|$)',text,re.M):
        return None
    return f"{heading}\n\n{text}"


_CONDENSE_SYSTEM_PROMPT = """You are cleaning up notes from ONE lecture recording session that were saved
multiple times (autosaves), which can leave duplicate or overlapping sections.

Rules:
- Merge into exactly ONE "## <date>" section - reuse the existing date, don't invent one
- Merge duplicate/overlapping bullets into a single clean bullet - don't just concatenate them
- CRITICAL: every bullet in the input covers a distinct piece of information UNLESS it is
  clearly restating the same fact as another bullet. Before finishing, check each input
  bullet individually and confirm it is represented in your output somewhere - do not
  drop a bullet just because it's short, or because it appeared in a section by itself
- Fix broken markdown formatting
- If a "### Q&A" section is present in the notes below, keep it separate, cleaned up the same way.
  If there is no Q&A content in the notes below, do not create one.
- Preserve any "[FLAG: ...]" or "verify" markers on the point they're attached to
- Do not invent new content or new sections
- Respond with ONLY the final markdown, no preamble or explanation, no notes about what you did"""


def condense_transcript(session_markdown: str) -> str | None:
    """Re-reviews notes from one lecture session (possibly containing duplicate/
    overlapping sections from multiple autosaves) and merges them into one clean
    section. Returns None if the local LLM server isn't available/fails. Currently
    unused (notes.py's condense_session is CLI/API-only - see its docstring for why),
    kept in sync with the same context/token-scaling fix as format_transcript in case
    that decision is ever revisited."""
    return _chat(_CONDENSE_SYSTEM_PROMPT, session_markdown, max_tokens=_scaled_max_tokens(session_markdown, ceiling=3000))
