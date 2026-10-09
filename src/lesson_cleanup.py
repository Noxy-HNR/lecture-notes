"""Cleanup of lesson build leftovers: the "Free up space" card on the Diagnostics page.

Everything here stays inside lessons/. state/ (the audio backups) and notes/ are never offered.
A finished lesson is only ever deleted when the student ticks it by name; nothing is suggested
for it. A running build owns its work folder, plan cache and job files, so none of them can be
touched until it ends.

scan() lists what could go. delete() scans again and acts only on items found in that fresh scan,
so a request can never name a path of its own: the ids it carries are looked up, never trusted.
"""
import contextlib
import json
import os
import re
import shutil
import stat
import time
from datetime import date, datetime
from pathlib import Path

import lessons

KINDS = ("partial", "plan", "job", "lesson")
MAX_DELETE_ITEMS = 500
PLAN_REUSE_SECONDS = 7 * 86400      # lessons.build_lesson ignores an older plan, so it is dead weight
FINISHED_JOB_SECONDS = 7 * 86400
FAILED_JOB_SECONDS = 86400

# What an id must look like for each kind. scan() only produces such ids, but delete() gets them
# from the network, so they are checked again before any path is built.
_ID_PATTERNS = {
    "partial": re.compile(r"[A-Za-z0-9_-]{1,120}\.partial"),
    "plan": re.compile(r"[A-Za-z0-9_-]{1,160}\.json"),
    "job": re.compile(r"[0-9a-f]{32}"),
    "lesson": re.compile(r"[A-Za-z0-9_-]{1,120}"),
}
# "CHEM_1450_20260916-20260928_<timestamp or plan digest>", the start of a lesson id and a plan name
_NAME_PARTS = re.compile(r"([A-Z]{2,5})_(\d{3,4}[A-Z]?)_(\d{8})-(\d{8})_")


class InUse(Exception):
    """A lesson is being built; its files must not be touched."""
    http_status = 409


# ---------------------------------------------------------------------------
# Reading the folder
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(lessons.LESSONS_DIR)      # looked up per call: tests point LESSONS_DIR elsewhere


def _is_link(path) -> bool:
    """A symlink or a Windows junction (is_symlink alone misses junctions)."""
    try:
        return Path(path).is_symlink() or Path(path).is_junction()
    except OSError:
        return True                        # can't tell, so leave it alone


def _entries(directory: Path) -> list[os.DirEntry]:
    """What is directly inside a folder, without links and without following one to get there."""
    if _is_link(directory):
        return []
    try:
        with os.scandir(directory) as scanned:
            return [entry for entry in scanned if not (entry.is_symlink() or entry.is_junction())]
    except OSError:
        return []


def _tree(path: Path) -> tuple[int, bool]:
    """(bytes, whether it holds a link) for a file or a whole folder. Links are never followed."""
    if path.is_file():
        try:
            return path.stat().st_size, False
        except OSError:
            return 0, False
    total, linked, pending = 0, False, [path]
    while pending:
        try:
            with os.scandir(pending.pop()) as scanned:
                for entry in scanned:
                    if entry.is_symlink() or entry.is_junction():
                        linked = True
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total, linked


def _builder_alive(job: dict) -> bool:
    """Whether a build record's builder plausibly still runs. When unsure it says yes: a build
    wrongly thought finished could lose its work folder, one wrongly thought busy only waits."""
    check = getattr(lessons, "_builder_alive", None)
    try:
        return bool(check(job)) if check else True
    except Exception:
        return True


def load_jobs() -> list[dict]:
    """Every build record in lessons/.jobs, newest first. Read-only: lessons.get_job would also
    rewrite a dead builder's record as failed, which a poll or a scan has no business doing.
    Each: {"id", "job" (the record, {} if unreadable), "status" (running | ready | failed |
    cancelled | unknown - a builder that died mid-build counts as failed), "running", "created",
    "modified" (newest file), "recorded" (the status file's time, when the state last changed),
    "bytes", "files"}."""
    directory = _root() / ".jobs"
    ids = {os.path.splitext(entry.name)[0] for entry in _entries(directory)
           if os.path.splitext(entry.name)[1] in (".json", ".log")
           and _ID_PATTERNS["job"].fullmatch(os.path.splitext(entry.name)[0])}
    jobs = []
    for job_id in ids:
        try:
            found = _job_record(directory, job_id)
        except OSError:
            continue                # deleted while it was being read: it isn't there any more
        if found is not None:
            jobs.append(found)
    return sorted(jobs, key=lambda found: found["created"], reverse=True)


def _job_record(directory: Path, job_id: str) -> dict | None:
    """One build's record as load_jobs lists it, or None when neither of its files is left."""
    files = [directory / f"{job_id}.json", directory / f"{job_id}.log"]
    files = [path for path in files if path.is_file() and not _is_link(path)]
    if not files:
        return None
    try:
        job = json.loads((directory / f"{job_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        job = {}
    job = job if isinstance(job, dict) else {}
    raw = job.get("status")
    running = raw == "running" and _builder_alive(job)
    status = ("running" if running else "failed" if raw == "running"
              else raw if raw in ("ready", "failed", "cancelled") else "unknown")
    sizes = [_tree(path)[0] for path in files]
    modified = max(path.stat().st_mtime for path in files)
    record = directory / f"{job_id}.json"
    # the status file is written when the build changes state; the log keeps growing after that
    recorded = record.stat().st_mtime if record in files else modified
    created = job.get("created_at")
    return {"id": job_id, "job": job, "status": status, "running": running,
            "created": created if isinstance(created, (int, float)) else modified,
            "modified": modified, "recorded": recorded, "bytes": sum(sizes), "files": files}


# ---------------------------------------------------------------------------
# Plain-English descriptions
# ---------------------------------------------------------------------------

def _day(timestamp: float) -> str:
    moment = datetime.fromtimestamp(timestamp)
    return f"{moment:%b} {moment.day}" + ("" if moment.year == datetime.now().year else f", {moment.year}")


def _iso_day(text) -> str | None:
    try:
        day = date.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return f"{day:%b} {day.day}"


def _span(start, end) -> str | None:
    first, last = _iso_day(start), _iso_day(end)
    if not first or not last:
        return None
    return first if first == last else f"{first} to {last}"


def _describe_name(name: str) -> str | None:
    """"CHEM 1450, Sep 16 to Sep 28" for a lesson id, work folder or plan name; None if it isn't one."""
    parts = _NAME_PARTS.match(name)
    if not parts:
        return None
    code, number, first, last = parts.groups()
    span = _span(*(f"{text[:4]}-{text[4:6]}-{text[6:]}" for text in (first, last)))
    return f"{code} {number}" + (f", {span}" if span else "")


def _lesson_details(path: Path, lesson_id: str) -> tuple[str, str]:
    """(label, detail) for a finished lesson, from its lesson.json."""
    try:
        record = json.loads((path / "lesson.json").read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("not a lesson")
    except (OSError, ValueError):
        return lesson_id, "Has no readable lesson.json, so it can't be played"
    parts = [str(record.get("title") or lesson_id)]
    code = record.get("class_code")
    if isinstance(code, str) and code:
        parts.append(code)
    span = _span(record.get("start"), record.get("end"))
    if span:
        parts.append(span)
    seconds = record.get("duration_seconds")
    if isinstance(seconds, (int, float)) and seconds >= 30:
        parts.append(f"{round(seconds / 60)} min")
    made = record.get("created_at")
    detail = "Finished lesson" + (f" from {_day(made)}" if isinstance(made, (int, float)) else "")
    return " · ".join(parts), detail + " - you can play it"


def _item(kind, name, label, detail, size, modified, lock=None, suggested=False) -> dict:
    return {"kind": kind, "id": name, "label": label, "detail": detail, "bytes": size,
            "modified": modified, "protected": lock is not None, "protected_reason": lock,
            "suggested": suggested and lock is None}


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def _build_lock(name: str, running: list[dict]) -> str | None:
    """Why a work folder or plan can't be touched right now, or None. While any build runs they are
    all kept: telling a leftover from the running build's own files isn't worth the risk."""
    if not running:
        return None
    for build in running:
        job = build["job"]
        code = str(job.get("class_code") or "").replace(" ", "_")
        prefix = f"{code}_{str(job.get('start')).replace('-', '')}-{str(job.get('end')).replace('-', '')}_"
        if code and name.startswith(prefix):
            return "Used by the lesson build running now"
    return "A lesson build is running - wait until it finishes"


def _partials(root: Path, running: list[dict]) -> list[dict]:
    items = []
    for entry in _entries(root):
        if not (entry.name.endswith(".partial") and entry.is_dir(follow_symlinks=False)
                and _ID_PATTERNS["partial"].fullmatch(entry.name)):
            continue
        try:
            modified = entry.stat().st_mtime
        except OSError:
            continue                            # removed since it was listed
        size, linked = _tree(Path(entry.path))
        lock = _build_lock(entry.name, running) or ("It holds a link, so it is left alone" if linked else None)
        what = _describe_name(entry.name)
        items.append(_item("partial", entry.name, "Unfinished lesson build" + (f": {what}" if what else ""),
                           f"Unfinished build folder from {_day(modified)}", size, modified, lock, suggested=True))
    return items


def _plans(root: Path, running: list[dict]) -> list[dict]:
    items = []
    for entry in _entries(root / ".plans"):
        if not (entry.is_file(follow_symlinks=False) and _ID_PATTERNS["plan"].fullmatch(entry.name)):
            continue
        try:
            stat_result = entry.stat()
        except OSError:
            continue                            # removed since it was listed
        modified = stat_result.st_mtime
        stale = time.time() - modified > PLAN_REUSE_SECONDS
        what = _describe_name(entry.name)
        detail = (f"Saved plan from {_day(modified)}, too old to be reused" if stale else
                  f"Saved plan from {_day(modified)}; a retry reuses it instead of writing the lesson again")
        items.append(_item("plan", entry.name, "Saved lesson plan" + (f": {what}" if what else ""), detail,
                           stat_result.st_size, modified, _build_lock(entry.name, running), suggested=stale))
    return items


def _job_item(build: dict) -> dict:
    job, status = build["job"], build["status"]
    what = ", ".join(part for part in (job.get("class_code"), _span(job.get("start"), job.get("end"))) if part)
    label = ("Lesson build record" + (f": {what}" if what else "")) if job else "Incomplete build record"
    if build["running"]:
        return _item("job", build["id"], label, "This build is running right now", build["bytes"],
                     build["modified"], lock="This build is running right now")
    outcome = {"ready": "Finished", "failed": "Failed", "cancelled": "Cancelled"}.get(status, "Unclear")
    age = time.time() - build["created"]
    old = age > (FINISHED_JOB_SECONDS if status == "ready" else FAILED_JOB_SECONDS)
    return _item("job", build["id"], label, f"{outcome} build from {_day(build['created'])}, with its log",
                 build["bytes"], build["modified"], suggested=old)


def _finished_lessons(root: Path) -> list[dict]:
    items = []
    for entry in _entries(root):
        if entry.name.startswith(".") or entry.name.endswith(".partial") \
                or not entry.is_dir(follow_symlinks=False) or not _ID_PATTERNS["lesson"].fullmatch(entry.name):
            continue
        path = Path(entry.path)
        if not (path / "lesson.json").is_file():
            continue                       # not a lesson: whatever it is, it isn't ours to offer
        try:
            modified = entry.stat().st_mtime
        except OSError:
            continue                            # removed since it was listed
        size, linked = _tree(path)
        label, detail = _lesson_details(path, entry.name)
        items.append(_item("lesson", entry.name, label, detail, size, modified,
                           "It holds a link, so it is left alone" if linked else None))
    return items


def scan() -> dict:
    """What lessons/ holds that could be deleted, grouped by kind and newest first. Read-only."""
    root = _root()
    builds = load_jobs()
    running = [build for build in builds if build["running"]]
    items = (_partials(root, running) + _plans(root, running) + [_job_item(build) for build in builds]
             + _finished_lessons(root))
    items.sort(key=lambda item: (KINDS.index(item["kind"]), -item["modified"]))
    # Protected items are counted too: "all" is everything listed, "suggested" what is ticked for you.
    return {"running_build": bool(running), "items": items,
            "totals": {"suggested_bytes": sum(item["bytes"] for item in items if item["suggested"]),
                       "all_bytes": sum(item["bytes"] for item in items)}}


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class _Refused(Exception):
    """An item that can't be deleted; the message is shown to the student."""


def _why(error: BaseException) -> str:
    """A reason for a student to read: no paths, and the usual Windows trouble in plain words."""
    if getattr(error, "winerror", None) == 32:
        return "a file in it is open in another program"
    if isinstance(error, PermissionError):
        return "Windows wouldn't allow it (a file may be in use)"
    return getattr(error, "strerror", None) or type(error).__name__


def _inside(path: Path, parent: Path) -> Path:
    """`path` itself, once it is certain to be a plain entry directly inside `parent`."""
    if _is_link(path) or _is_link(parent):
        raise _Refused("it is a link, so it is left alone")
    if path.resolve().parent != parent.resolve():
        raise _Refused("it isn't where lesson files are kept")
    return path


def _remove_file(path: Path) -> tuple[int, str | None]:
    try:
        size = path.stat().st_size
        path.unlink()
        return size, None
    except FileNotFoundError:
        return 0, None
    except OSError as error:
        return 0, _why(error)


def _stuck(reason: str | None) -> str | None:
    """How a folder that could not be removed completely is reported."""
    return None if reason is None else f"couldn't remove all of it: {reason}"


def _remove_tree(path: Path) -> tuple[int, str | None]:
    """Deletes a folder. A file that can't go (open in the player, say) is recorded, not raised, and
    the rest of the folder is still removed. Returns (bytes freed, why something is left or None)."""
    before, _ = _tree(path)
    errors = []

    def on_error(function, failed, error):
        if isinstance(error, PermissionError) and getattr(error, "winerror", None) != 32:
            try:
                os.chmod(failed, stat.S_IWRITE)   # a read-only file; an in-use one fails again below
                function(failed)
                return
            except OSError as again:
                error = again
        errors.append(error)

    try:
        shutil.rmtree(path, onexc=on_error)
    except OSError as error:
        errors.append(error)
    if not os.path.lexists(path):
        return before, None
    after, _ = _tree(path)
    return max(before - after, 0), _why(errors[0]) if errors else "it is still there"


def _remove_lesson(folder: Path) -> tuple[int, str | None]:
    """A finished lesson, removed in two steps. lesson.json is what makes a folder a lesson (scan and the
    Lessons page look for it), so it goes last: while anything else is stuck the lesson stays listed and
    can be tried again, rather than its audio being left behind where nothing can find it."""
    try:
        with os.scandir(folder) as scanned:
            entries = [entry for entry in scanned if entry.name != "lesson.json"]
    except OSError as error:
        return 0, _why(error)
    freed, stuck = 0, []
    for entry in entries:
        if entry.is_symlink() or entry.is_junction():
            # never followed, and never dropped quietly: that would let lesson.json go while it is still here
            stuck.append("it holds a link, so it is left alone")
            continue
        if entry.is_dir(follow_symlinks=False):
            size, reason = _remove_tree(Path(entry.path))
        else:
            size, reason = _remove_file(Path(entry.path))
        freed += size
        if reason:
            stuck.append(reason)
    if stuck:
        return freed, stuck[0]
    size, reason = _remove_file(folder / "lesson.json")
    freed += size
    if reason:
        return freed, reason
    size, reason = _remove_tree(folder)            # the emptied folder
    return freed + size, reason


def _remove(root: Path, kind: str, name: str) -> tuple[int, str | None]:
    """Deletes one scanned item; returns (bytes freed, why it wasn't completely deleted or None)."""
    if kind == "job":
        directory = root / ".jobs"
        freed = 0
        for suffix in (".log", ".json"):     # the record last: if the log is stuck it stays, whole, to retry
            size, error = _remove_file(_inside(directory / f"{name}{suffix}", directory))
            freed += size
            if error:
                return freed, error
        return freed, None
    if kind == "plan":
        directory = root / ".plans"
        return _remove_file(_inside(directory / name, directory))
    if kind == "lesson":
        freed, problem = _remove_lesson(_inside(root / name, root))
        if problem is None:
            quiz = root / ".quiz"      # what the student answered in this lesson goes with it
            if not _is_link(quiz):
                _remove_file(quiz / f"{name}.json")
        return freed, _stuck(problem)
    freed, problem = _remove_tree(_inside(root / name, root))
    return freed, _stuck(problem)


def _reference(entry) -> dict:
    """Echoes a requested item in a reply without trusting its shape."""
    if isinstance(entry, dict):
        return {"kind": str(entry.get("kind"))[:20], "id": str(entry.get("id"))[:80]}
    return {"kind": "?", "id": str(entry)[:80]}


def delete(payload: dict) -> dict:
    """Deletes the ticked items for good (no recycle bin). Needs `confirm` to be exactly true.
    Bad, stale, protected or stubborn items are skipped with a reason and the rest still go."""
    if not isinstance(payload, dict) or payload.get("confirm") is not True:
        raise ValueError("Deleting needs your confirmation")
    wanted = payload.get("items")
    if not isinstance(wanted, list):
        raise ValueError("Say which items to delete")
    if len(wanted) > MAX_DELETE_ITEMS:
        raise ValueError(f"Delete at most {MAX_DELETE_ITEMS} items at a time")
    root = _root()
    deleted, skipped, freed, seen = [], [], 0, set()
    # Holding the lock a build is started under means none can begin between the scan and the deletes.
    with getattr(lessons, "_jobs_lock", None) or contextlib.nullcontext():
        current = scan()
        known = {(item["kind"], item["id"]): item for item in current["items"]}
        if current["running_build"] and any(isinstance(entry, dict) and entry.get("kind") in ("partial", "plan")
                                            for entry in wanted):
            raise InUse("A lesson is being built - wait until it finishes.")
        for entry in wanted:
            reference = _reference(entry)
            if isinstance(entry, dict) and isinstance(entry.get("kind"), str) and isinstance(entry.get("id"), str):
                key = (entry["kind"], entry["id"])
            else:
                skipped.append({"item": reference, "reason": "That isn't a valid item"})
                continue
            if key in seen:
                continue
            seen.add(key)
            pattern = _ID_PATTERNS.get(key[0])
            if pattern is None or not pattern.fullmatch(key[1]):
                skipped.append({"item": reference, "reason": "That isn't a valid item name"})
            elif key not in known:
                skipped.append({"item": reference, "reason": "It is already gone, or isn't something that can be cleaned up"})
            elif known[key]["protected"]:
                skipped.append({"item": reference, "reason": known[key]["protected_reason"]})
            else:
                try:
                    size, problem = _remove(root, *key)
                except _Refused as refusal:
                    size, problem = 0, str(refusal)
                except OSError as error:
                    size, problem = 0, _why(error)
                freed += size
                if problem:
                    skipped.append({"item": reference, "reason": problem[0].upper() + problem[1:]})
                else:
                    deleted.append({"kind": key[0], "id": key[1]})
    return {"deleted": deleted, "skipped": skipped, "freed_bytes": freed}
