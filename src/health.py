"""System health for the Diagnostics page: whether Claude can be reached, how much RAM and disk are
free, and why the last lesson build failed.

Written after every lesson build failed for days with "Claude CLI/API is unavailable" while the
real cause - a 10-minute timeout - was only visible by digging through files. The page polls
snapshot(), so it is cheap, changes nothing and never calls a model; every section catches its own
errors, so one broken probe never hides the rest. test_claude() is the one call that uses Claude,
and it only runs when the student presses "Test Claude".

Memory is read the way C:/AI/AGENTS.md asks: available physical RAM (ullAvailPhys), not "free" RAM.
"""
import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "notes"
LESSONS_DIR = ROOT / "lessons"
STATE_DIR = ROOT / "state"           # also holds health_claude.json, the last "Test Claude" result

GIB = 1024 ** 3
RAM_LOW_GB, RAM_CRITICAL_GB = 2.0, 1.0
DISK_LOW_GB, DISK_CRITICAL_GB = 5.0, 1.0
RECORDER_STALE_SECONDS = 30          # the Diagnostics banner's own "no heartbeat" rule
LLAMA_PORT = 10000                   # the local model server the C:/AI dashboard manages; only ever probed
VERSION_TIMEOUT_SECONDS = 5
STORAGE_BUDGET_SECONDS = 1.5         # state/ is ~17 GB in many files; a poll must never wait long for it
TEST_PROMPT = "Reply with exactly: OK"
TEST_TIMEOUT_SECONDS = 45
MIN_TEST_GAP_SECONDS = 10
OLD_TEST_SECONDS = 86400             # a failed test older than this is history, not a current problem

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _now() -> float:
    return time.time()


def _clock() -> float:
    return time.monotonic()


class Busy(Exception):
    """A Claude test is running or was only just done."""
    http_status = 409


# ---------------------------------------------------------------------------
# Caching the slow probes
# ---------------------------------------------------------------------------

class _Cached:
    """One slow probe's result, kept for `ttl` seconds. A poll that arrives while another one is
    computing gets the last value instead of starting a second probe (two tabs poll at once)."""

    def __init__(self, ttl: float):
        self.ttl = ttl
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.key, self.value, self.at = None, None, None

    def _fresh(self, key) -> bool:
        return self.at is not None and self.key == key and _clock() - self.at < self.ttl

    def get(self, key, compute):
        if self._fresh(key):
            return self.value
        if not self._lock.acquire(blocking=self.at is None):   # nothing to fall back on the first time
            return self.value
        try:
            if not self._fresh(key):
                self.value, self.key, self.at = compute(), key, _clock()
            return self.value
        finally:
            self._lock.release()


_VERSION = _Cached(60)
_STORAGE = _Cached(60)
_NARRATION = _Cached(300)            # the first speech.available() imports onnxruntime; don't repeat it
_LID = _Cached(float("inf"))         # whether the PC reports a lid doesn't change while this runs
_CACHES = (_VERSION, _STORAGE, _NARRATION, _LID)


def reset_caches():
    """Forgets every cached probe and the last Claude test held in memory (for tests)."""
    global _last_result, _result_loaded, _last_started
    for cache in _CACHES:
        cache.reset()
    _last_result, _result_loaded, _last_started = None, False, None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _probe_version(path: str) -> str | None:
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, timeout=VERSION_TIMEOUT_SECONDS,
                              creationflags=_NO_WINDOW)
    except Exception:
        return None              # a failure is cached like a success, so a hung CLI costs one wait a minute
    lines = (done.stdout or "").strip().splitlines()
    return lines[0][:80] if done.returncode == 0 and lines else None


def _claude() -> dict:
    import notes
    path = notes._find_claude_cli()
    return {"cli_found": bool(path), "cli_path": path or None,
            "version": _VERSION.get(path, lambda: _probe_version(path)) if path else None,
            # whether the key exists, never what it is
            "api_key": "set" if os.environ.get("ANTHROPIC_API_KEY") else "not set",
            "last_test": _last_test()}


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _read_memory() -> tuple[int, int]:
    """(available, total) physical RAM in bytes."""
    if sys.platform != "win32":
        raise OSError("RAM can only be read on Windows")
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.ullAvailPhys, status.ullTotalPhys


def _level(free_gb: float, low: float, critical: float) -> str:
    return "critical" if free_gb < critical else "low" if free_gb < low else "ok"


def _memory() -> dict:
    available, total = _read_memory()
    free_gb = available / GIB
    return {"available_gb": round(free_gb, 2), "total_gb": round(total / GIB, 2),
            "level": _level(free_gb, RAM_LOW_GB, RAM_CRITICAL_GB)}


def _disk() -> dict:
    usage = shutil.disk_usage(ROOT)       # the drive that holds the project
    free_gb = usage.free / GIB
    return {"free_gb": round(free_gb, 1), "total_gb": round(usage.total / GIB, 1),
            "level": _level(free_gb, DISK_LOW_GB, DISK_CRITICAL_GB)}


def _recorder() -> dict:
    import telemetry
    status = telemetry.read_status()
    updated = status.get("updated_at")
    fresh = isinstance(updated, (int, float)) and _now() - updated <= RECORDER_STALE_SECONDS
    active = bool(status.get("active")) and fresh
    return {"active": active, "class_code": (status.get("class_code") or None) if active else None}


# Where a build was when it stopped, from the progress lessons.build_lesson reports on the way.
# A failed build's own stage reads just "Failed", so the step is worked out from how far it got.
_STEPS = ((0.08, "Reading the notes"), (0.25, "Writing the lesson"), (0.35, "Checking the lesson length"),
          (0.45, "Finding pictures"), (0.5, "Choosing the best pictures"))
_END_STAGES = ("Failed", "Cancelled", "Canceled", "Cancelling", "Starting")


def _step(job: dict) -> str | None:
    steps = job.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[-1], dict) and isinstance(steps[-1].get("name"), str):
        return steps[-1]["name"]          # the step the build was in when it stopped, as lessons recorded it
    stage = job.get("stage")
    if isinstance(stage, str) and stage and stage not in _END_STAGES:
        return stage
    progress = job.get("progress")
    if not isinstance(progress, (int, float)):
        return None
    return next((name for limit, name in _STEPS if progress < limit), "Recording narration")


def _log_tail(path: Path, size: int = 2048) -> str:
    """The last line of a build's log - a traceback ends on the exception - without reading all of it."""
    with open(path, "rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(log.tell() - size, 0))
        lines = [line.strip() for line in log.read().decode("utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def _last_log_line(files: list[Path]) -> str:
    for path in files:
        if path.suffix == ".log":
            try:
                tail = _log_tail(path)
            except OSError:
                continue
            if tail:
                return tail
    return ""


def _failure_reason(build: dict) -> str:
    error = build["job"].get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()[:300]
    if build["status"] == "cancelled":
        return "Cancelled before it finished"      # whatever the log says, the student stopped it
    tail = _last_log_line(build["files"])
    if build["job"].get("status") == "running":     # the builder died without saying why: the log line is only a clue
        return "The lesson builder stopped unexpectedly" + (f" (last log line: {tail})" if tail else "")
    return tail or "No reason was recorded"


def _range(job: dict) -> str | None:
    start, end = job.get("start"), job.get("end")
    if not start or not end:
        return None
    return str(start) if start == end else f"{start} to {end}"


def _lesson_build() -> dict:
    import lesson_cleanup
    jobs = lesson_cleanup.load_jobs()
    running = next((build for build in jobs if build["running"]), None)
    failed = next((build for build in jobs if build["status"] in ("failed", "cancelled")), None)
    summary = failure = None
    if running:
        job = running["job"]
        summary = {"id": running["id"], "class_code": job.get("class_code"), "range": _range(job),
                   "stage": job.get("stage"), "progress": job.get("progress"), "started": running["created"]}
    if failed:
        job = failed["job"]
        failure = {"when": failed["recorded"], "class_code": job.get("class_code"), "range": _range(job),
                   "step": _step(job), "reason": _failure_reason(failed), "status": failed["status"],
                   # a later build that finished or is running makes this old news
                   "superseded": any(build["created"] > failed["created"] and build["status"] in ("ready", "running")
                                     for build in jobs)}
    return {"running": summary, "last_failure": failure}


def _narration() -> dict:
    def probe():
        try:
            import speech
            usable, reason = speech.available()
            return {"available": bool(usable), "reason": str(reason or "")}
        except Exception as error:
            return {"available": False, "reason": f"couldn't check ({type(error).__name__}: {error})"}
    return dict(_NARRATION.get(None, probe))


def _lid_watch() -> dict:
    """Whether Windows reports a laptop lid. The recorder owns the real watcher; this starts one,
    stops it at once and remembers the answer, because every start registers a window class that is
    never unregistered."""
    def probe():
        try:
            import lid_watch
            watcher = lid_watch.LidWatcher(lambda: None)
            try:
                return bool(watcher.start(timeout=1.0))
            finally:
                watcher.stop()
        except Exception:
            return False
    return {"available": _LID.get(None, probe)}


def _llama_server() -> dict:
    """Whether something listens on the local model server's port. Looking only: the C:/AI dashboard
    is the one thing allowed to start or stop it."""
    try:
        with socket.create_connection(("127.0.0.1", LLAMA_PORT), timeout=0.3):
            return {"listening": True, "port": LLAMA_PORT}
    except OSError:
        return {"listening": False, "port": LLAMA_PORT}


def _walk_bytes(path: Path, deadline: float) -> tuple[int, bool]:
    """(bytes under path, whether the walk finished before the deadline). Links are not followed."""
    total, pending = 0, [path]
    while pending:
        try:
            with os.scandir(pending.pop()) as scanned:
                for entry in scanned:
                    # every entry, not every few hundred: a folder of many small files must not
                    # outrun the budget between two looks at the clock
                    if _clock() > deadline:
                        return total, False
                    if entry.is_symlink() or entry.is_junction():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        if _clock() > deadline:
            return total, False
    return total, True


def _storage() -> dict:
    folders = (NOTES_DIR, LESSONS_DIR, STATE_DIR)

    def measure():
        deadline = _clock() + STORAGE_BUDGET_SECONDS
        sizes, finished = [], True
        for folder in folders:
            if not finished:
                sizes.append(None)       # out of time before reaching it
                continue
            total, finished = _walk_bytes(folder, deadline)
            sizes.append(total)
        notes, lessons, state = sizes
        return {"notes_mb": None if notes is None else round(notes / 1024 ** 2, 1),
                "lessons_mb": None if lessons is None else round(lessons / 1024 ** 2, 1),
                "state_gb": None if state is None else round(state / GIB, 2),
                "approximate": not finished}
    return dict(_STORAGE.get(folders, measure))


# ---------------------------------------------------------------------------
# Test Claude
# ---------------------------------------------------------------------------

_test_lock = threading.Lock()
_last_result = None              # {"ok", "seconds", "reason", "at"} of the last test, or None
_result_loaded = False           # state/health_claude.json is read once, so it survives a restart
_last_started = None


def _test_file() -> Path:
    return STATE_DIR / "health_claude.json"


def _load_test_file() -> dict | None:
    try:
        data = json.loads(_test_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (isinstance(data, dict) and isinstance(data.get("ok"), bool)
            and isinstance(data.get("seconds"), (int, float)) and isinstance(data.get("at"), (int, float))
            and (data.get("reason") is None or isinstance(data.get("reason"), str))):
        return {key: data[key] for key in ("ok", "seconds", "reason", "at")}
    return None


def _last_test() -> dict | None:
    global _last_result, _result_loaded
    if _last_result is None and not _result_loaded:
        _result_loaded = True
        _last_result = _load_test_file()
    return dict(_last_result) if _last_result else None


def _save_test(result: dict):
    try:
        STATE_DIR.mkdir(exist_ok=True)
        tmp = _test_file().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result), encoding="utf-8")
        os.replace(tmp, _test_file())      # atomic: a poll never reads half a file
    except OSError:
        pass                               # the result is still in memory and in the reply


def test_claude(payload=None) -> dict:
    """One tiny real call to Claude through the CLI, so the student can tell whether lessons and notes
    can reach it. Uses a few seconds of Claude; only runs when its button is pressed."""
    global _last_result, _result_loaded, _last_started
    if not _test_lock.acquire(blocking=False):
        raise Busy("Claude is already being tested - wait for it to finish")
    try:
        if _last_started is not None and _clock() - _last_started < MIN_TEST_GAP_SECONDS:
            raise Busy("Claude was only just tested - wait a few seconds and try again")
        _last_started = started = _clock()
        import notes
        try:
            reply = notes._try_claude_cli_format(TEST_PROMPT, timeout=TEST_TIMEOUT_SECONDS)
            reason = None if reply else (notes.last_cli_failure or "the Claude CLI gave no reply")
        except Exception as error:
            reply, reason = None, f"the Claude CLI could not be run ({type(error).__name__}: {error})"
        if reason:
            # notes words a timeout in whole minutes, which reads "within 0 minutes" for 45 seconds
            reason = reason.replace("within 0 minutes", f"within {TEST_TIMEOUT_SECONDS} seconds")
        result = {"ok": bool(reply), "seconds": round(_clock() - started, 1), "reason": reason, "at": _now()}
        _last_result, _result_loaded = result, True
        _save_test(result)
        return dict(result)
    finally:
        _test_lock.release()


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

_SECTIONS = {"claude": _claude, "memory": _memory, "disk": _disk, "recorder": _recorder,
             "lesson_build": _lesson_build, "narration": _narration, "lid_watch": _lid_watch,
             "llama_server": _llama_server, "storage": _storage}
_SECTION_NAMES = {"claude": "Claude", "memory": "RAM", "disk": "disk space", "recorder": "the recorder",
                  "lesson_build": "lesson builds", "narration": "narration", "lid_watch": "the lid watch",
                  "llama_server": "the local model server", "storage": "storage sizes"}


def _guarded(probe) -> dict:
    try:
        return probe()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:200]}


def _problems(sections: dict) -> list[str]:
    """What is wrong right now, most urgent first, in words a student can act on."""
    def ok(name):
        found = sections.get(name)
        return found if isinstance(found, dict) and "error" not in found else {}

    found = []
    memory = ok("memory")
    if memory.get("level") == "critical":
        found.append("Less than 1 GB of RAM free - do not start a lesson build")
    elif memory.get("level") == "low":
        found.append(f"Only {memory['available_gb']:.1f} GB of RAM is free - close some programs before building a lesson")
    claude = ok("claude")
    if claude and not claude.get("cli_found"):
        found.append("Claude CLI not found - " + ("the API key will be used instead"
                     if claude.get("api_key") == "set" else "and no ANTHROPIC_API_KEY is set, so lessons can't be written"))
    test = claude.get("last_test")
    # without the CLI the test can't run at all, and the line above already says so
    if test and not test.get("ok") and claude.get("cli_found") and _now() - test.get("at", 0) < OLD_TEST_SECONDS:
        found.append(f"The last Claude test failed: {test.get('reason') or 'no reason was given'}")
    failure = ok("lesson_build").get("last_failure")
    if failure and failure.get("status") == "failed" and not failure.get("superseded"):
        where = f" at {failure['step']}" if failure.get("step") else ""
        found.append(f"The last lesson build failed{where}: {str(failure.get('reason')).rstrip('.')}")
    disk = ok("disk")
    if disk.get("level") == "critical":
        found.append("Less than 1 GB of disk space is free - recordings may not save")
    elif disk.get("level") == "low":
        found.append(f"Only {disk['free_gb']:.1f} GB of disk space is free")
    narration = ok("narration")
    if narration and not narration.get("available"):
        found.append(f"Lessons will have no narration: {narration.get('reason') or 'the voice is not ready'}")
    for name, label in _SECTION_NAMES.items():
        error = sections.get(name, {}).get("error") if isinstance(sections.get(name), dict) else None
        if error:
            found.append(f"Couldn't check {label}: {error}")
    return found


def snapshot() -> dict:
    """Everything the health card shows. Cheap: slow probes are cached, nothing is started or changed."""
    sections = {name: _guarded(probe) for name, probe in _SECTIONS.items()}
    sections["problems"] = _guarded_list(lambda: _problems(sections))
    sections["checked_at"] = _now()
    return sections


def _guarded_list(build) -> list[str]:
    try:
        return build()
    except Exception as error:
        return [f"Couldn't work out what is wrong: {type(error).__name__}: {error}"[:200]]
