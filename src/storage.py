"""Atomic UTF-8 writes with durable, timestamped revisions of existing content."""
import os
import tempfile
from datetime import datetime
from pathlib import Path
import re
import time
from contextlib import contextmanager


class RevisionConflict(ValueError):
    """The file changed after the caller read it."""


@contextmanager
def file_lock(path, timeout=10):
    """Coordinate read/modify/write across the recorder and dashboard processes."""
    path = Path(path)
    lock_dir = path.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / (path.name + ".lock")).open("a+b") as handle:
        handle.seek(0,2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("File is busy; try saving again")
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def atomic_write(path, text, backup=True, expected=None):
    with file_lock(path):
        _atomic_write(path,text,backup,expected)


def _atomic_write(path, text, backup=True, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if expected is not None and current != expected:
        raise RevisionConflict("This file changed since you opened it. Reload before saving.")
    if backup and path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous == text:
            return
        revisions = path.parent / ".revisions" / path.name
        revisions.mkdir(parents=True, exist_ok=True)
        revision = revisions / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".txt")
        atomic_write(revision, previous, backup=False)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def session_markers(session_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise ValueError("Invalid session ID")
    return f"<!-- session:{session_id} -->", f"<!-- /session:{session_id} -->"


def session_bounds(content, session_id):
    start, end = session_markers(session_id)
    left = content.find(start)
    if left < 0:
        return None
    right = content.find(end, left + len(start))
    if right < 0:
        raise ValueError("Incomplete session section; restore a revision before saving")
    return left, right + len(end)


def update_session(content, session_id, section, replace=False):
    start, end = session_markers(session_id)
    section = re.sub(r"<!-- /?session:[A-Za-z0-9_-]+ -->", "", section)
    bounds = session_bounds(content, session_id)
    if bounds:
        left, right = bounds
        prior = content[left + len(start):right - len(end)].strip()
        body = section.strip() if replace else prior + "\n\n" + section.strip()
        return content[:left] + start + "\n" + body + "\n" + end + content[right:]
    return content.rstrip() + "\n\n" + start + "\n" + section.strip() + "\n" + end + "\n"
