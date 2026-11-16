"""Fixes lecture notes after the fact: rough formatting, uncertain spots, the wrong class.

Three things go wrong with notes, and this module repairs each one:

  rough formatting   A save that happened while Claude was unreachable fell back to the
                     heuristic formatter: bullets that are close to the raw transcript.
                     `reformat` rebuilds that lecture from its saved transcript with Claude.
  uncertain spots    Claude marks mis-heard words "⚠️ verify". `resolve` asks Claude to work
                     out what was meant from context and the subject, and remove the marks.
  wrong class        class_match.py notices a recording that sounds like another class;
                     `move_session` moves its notes and files there.

They run as one background job at a time (state/repair_job.json), started from the
dashboard's Cleanup view or automatically:
  - after each lecture, the recorder starts `repair.py --auto --lecture <code> <session>`:
    that lecture gets both fixes, any other rough lecture is re-formatted, and finished
    recordings are compressed (archive.py)
  - when the dashboard starts, `repair.py --auto` catches up on anything Claude missed

Every write checks that the notes didn't change underneath it, keeps the previous version
in notes/.revisions (storage.atomic_write), and is refused while a lecture is recording.
Claude only ever writes here through the subscription CLI (or the API key, if set); the
local model is never used for repairs, since its result would only be thrown away.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import notes  # noqa: E402
from storage import atomic_write, session_markers, update_session, dedupe_day_headings  # noqa: E402
from transcripts import load_transcript, session_class  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "notes"
STATE_DIR = ROOT / "state"
JOB_PATH = STATE_DIR / "repair_job.json"
MEMORY_PATH = STATE_DIR / "repair_memory.json"

# What counts as an uncertain spot: the formatter's "⚠️ verify" and the proofreader's
# "[FLAG: ...]", plus the hedges Claude writes around them ("in the transcript", "unclear").
UNCERTAIN = re.compile(r"⚠️|\[FLAG|verify\*|\(verify|transcript\)|in the transcript|the transcript says|unclear",
                       re.I)
MARKERS = ("⚠️", "[FLAG")  # a resolved section must contain none of these
RESOLVE_MIN_SHARE = 0.55     # resolving trims hedges; losing more than this means content went
RESOLVE_MAX_SHARE = 1.3      # ...and it must not grow into an essay either
RESOLVE_CHECK_MIN_WORDS = 80
RESOLVE_TIMEOUT_SECONDS = 600
FAILED_RETRY_SECONDS = 7 * 86400  # an automatic job won't retry the same failed text sooner
RECORDING_FRESH_SECONDS = 120

_SESSION_BLOCK = re.compile(r"<!-- session:([A-Za-z0-9_-]+) -->\n?(.*?)\n?<!-- /session:\1 -->", re.S)
_DAY = re.compile(r"^## +(.+?)\s*$", re.M)
_job_lock = threading.Lock()


class RepairBusy(RuntimeError):
    http_status = 409


class RepairRefused(RuntimeError):
    http_status = 409


# ---------------------------------------------------------------------------
# Reading the notes
# ---------------------------------------------------------------------------

def notes_file(code: str) -> Path:
    path = notes.notes_path(code)
    if path.parent.resolve() != NOTES_DIR.resolve() or path.stem.endswith("_study_guide"):
        raise ValueError("Unknown class")
    return path


def class_codes() -> list[str]:
    return sorted(p.stem.replace("_", " ") for p in NOTES_DIR.glob("*.md") if not p.stem.endswith("_study_guide"))


def class_title(code: str, content: str | None = None) -> str:
    if content is None:
        path = notes_file(code)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    match = re.match(r"^# (.+?)\s*\(", content)
    if match:
        return match.group(1).strip()
    try:
        import schedule as sched
        cls = sched.get_class_by_code(code)
        return cls["title"] if cls else code
    except Exception:
        return code


def _uncertain_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if UNCERTAIN.search(line))


def _date_from_session(session_id: str) -> str:
    stamp = datetime.strptime(session_id[-15:], "%Y%m%d_%H%M%S")
    return f"{stamp.strftime('%A, %B')} {stamp.day}, {stamp.year}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_units(content: str) -> list[dict]:
    """Splits a notes file into the pieces repairs work on, in file order:
    a recorded session's block ("session:<id>"), or a dated section written before sessions
    were marked ("day:<heading>"). Each has the exact character span of its body."""
    units = []
    title_end = 0
    if content.startswith("# "):
        split = content.find("\n")
        title_end = split + 1 if split >= 0 else len(content)

    def loose(start, end):
        """Dated sections in text outside any session block."""
        piece = content[start:end]
        heads = list(_DAY.finditer(piece))
        for i, head in enumerate(heads):
            body_end = heads[i + 1].start() if i + 1 < len(heads) else len(piece)
            body = piece[head.start():body_end]
            if not body.strip() or not body[head.end() - head.start():].strip():
                continue
            units.append({"kind": "day", "day": head.group(1).strip(),
                          "start": start + head.start(), "end": start + body_end})

    cursor = title_end
    for match in _SESSION_BLOCK.finditer(content):
        loose(cursor, match.start())
        units.append({"kind": "session", "session_id": match.group(1),
                      "start": match.start(2), "end": match.end(2)})
        cursor = match.end()
    loose(cursor, len(content))

    seen = {}
    for unit in units:
        body = content[unit["start"]:unit["end"]]
        if unit["kind"] == "session":
            own = _DAY.findall(body)
            day = own[0].strip() if own else None
            if day is None:
                above = _DAY.findall(content[:unit["start"]])
                day = above[-1].strip() if above else _date_from_session(unit["session_id"])
            unit["day"] = day
            unit["key"] = "session:" + unit["session_id"]
        else:
            n = seen.get(unit["day"], 0)
            seen[unit["day"]] = n + 1
            unit["key"] = "day:" + unit["day"] + (f"#{n + 1}" if n else "")
        unit["text"] = body
    return units


def find_unit(content: str, key: str) -> dict:
    for unit in parse_units(content):
        if unit["key"] == key:
            return unit
    raise FileNotFoundError("That lecture isn't in the notes any more")


def session_transcript(session_id: str) -> str | None:
    try:
        segments = load_transcript(STATE_DIR, session_id)["segments"]
    except (FileNotFoundError, ValueError):
        return None
    text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
    return text or None


def overview(code: str) -> dict:
    """What the dashboard's Cleanup view shows for one class: every lecture with what's wrong
    with it, plus the job that's running (if any)."""
    path = notes_file(code)
    if not path.exists():
        raise FileNotFoundError(f"No notes for {code}")
    content = path.read_text(encoding="utf-8")
    rows = []
    for unit in parse_units(content):
        body = unit["text"]
        row = {"key": unit["key"], "kind": unit["kind"], "day": unit["day"],
               "words": len(body.split()), "rough": notes.HEURISTIC_MARKER in body,
               "uncertain": _uncertain_lines(body), "has_transcript": False, "suggest": None,
               "session_id": unit.get("session_id"), "time": None}
        if unit["kind"] == "session":
            sid = unit["session_id"]
            row["time"] = f"{sid[-6:-4]}:{sid[-4:-2]}"
            transcript = session_transcript(sid)
            row["has_transcript"] = transcript is not None
            if transcript:
                try:
                    import class_match
                    row["suggest"] = class_match.suggest(transcript, code, exclude_session=sid)
                except Exception:
                    row["suggest"] = None
        rows.append(row)
    return {"class_code": code, "title": class_title(code, content), "units": rows,
            "classes": class_codes(), "job": current_job(), "recording": recording_active(),
            "claude": notes.cli_available() or bool(os.environ.get("ANTHROPIC_API_KEY"))}


def recording_active() -> bool:
    try:
        import telemetry
        status = telemetry.read_status()
    except Exception:
        return False
    return bool(status.get("active")) and time.time() - float(status.get("updated_at") or 0) < RECORDING_FRESH_SECONDS


def _refuse_while_recording():
    if recording_active():
        raise RepairRefused("A lecture is recording right now. Notes are only repaired once it ends.")


def _write(code: str, content: str, updated: str):
    updated = dedupe_day_headings(updated)
    atomic_write(notes_file(code), updated, expected=content)


# ---------------------------------------------------------------------------
# The three repairs
# ---------------------------------------------------------------------------

def reformat(code: str, session_id: str) -> dict:
    """Rebuilds one recorded lecture's notes from its transcript with Claude. Existing notes are
    only replaced by Claude's; if Claude can't be reached they stay as they are."""
    _refuse_while_recording()
    path = notes_file(code)
    content = path.read_text(encoding="utf-8")
    unit = find_unit(content, "session:" + session_id)
    transcript = session_transcript(session_id)
    if not transcript:
        return {"ok": False, "message": "This lecture has no saved transcript to rebuild it from"}
    _, method = notes.format_and_save(code, class_title(code, content), transcript, unit["day"],
                                      mode="auto", combine_proofread=True, session_id=session_id,
                                      replace_session=True, existing_method="cli")
    if method == notes.KEPT_EXISTING:
        return {"ok": False, "claude_down": True,
                "message": "Claude couldn't be reached" + (f" ({notes.last_cli_failure})" if notes.last_cli_failure else "")
                           + ", so the notes were left as they were"}
    return {"ok": True, "message": "Re-formatted from the transcript"}


def _resolve_prompt(title: str, code: str, text: str) -> str:
    return f"""You are finalizing study notes for {title} ({code}), a college course. The notes were
made from automatic lecture transcripts, so they contain uncertain spots: "⚠️ verify" markers,
"[FLAG: ...]" markers, garbled words quoted "in the transcript", and remarks that something is
"unclear".

Resolve EVERY one of them:
- Work out what the lecturer actually meant from the surrounding context and standard knowledge
  of the subject at this level, and write the correct term or fact plainly (e.g. a garbled
  "unicyclic recessariums" in a biology lecture -> unicellular eukaryotes).
- Remove the hedging entirely: no ⚠️ markers, no [FLAG] markers, no quoted garbled transcript
  words, no notes like "likely a mis-transcription", "meaning unclear", "per the transcript".
- If a point states something factually wrong for the subject, correct it.
- If a fragment can't be reconstructed into anything meaningful, delete it rather than guessing wildly.
- Keep everything else as it is: the same headings (copy every "## " date heading exactly), order,
  bullets, bold terms and level of detail. Don't add new sections or outside material beyond
  what is needed to fill in a gap correctly.
- Plain markdown, no LaTeX.
- Output ONLY the finished markdown, no preamble or explanation.

Notes:
---
{text}
---
"""


def resolve_problem(original: str, resolved: str) -> str | None:
    """Why a resolved section can't replace the original, or None when it can."""
    first = next((line for line in resolved.splitlines() if line.strip()), "")
    if not notes._NOTES_LINE_START.match(first):
        return "the reply started with a message instead of notes"
    if any(marker in resolved for marker in MARKERS):
        return "it still had uncertain markers"
    if [d.strip() for d in _DAY.findall(resolved)] != [d.strip() for d in _DAY.findall(original)]:
        return "it changed the lecture's date heading"
    words, new_words = len(original.split()), len(resolved.split())
    if words >= RESOLVE_CHECK_MIN_WORDS and not RESOLVE_MIN_SHARE <= new_words / words <= RESOLVE_MAX_SHARE:
        return f"it came back {new_words} words long from {words}"
    return None


def resolve(code: str, key: str) -> dict:
    """Clears one lecture's uncertain spots with Claude."""
    _refuse_while_recording()
    path = notes_file(code)
    content = path.read_text(encoding="utf-8")
    unit = find_unit(content, key)
    original = unit["text"].strip()
    if not _uncertain_lines(original):
        return {"ok": True, "message": "Nothing uncertain left"}
    prompt = _resolve_prompt(class_title(code, content), code, original)
    problem = "Claude couldn't be reached"
    for attempt in range(2):
        reply = notes._try_claude_cli_format(prompt, timeout=RESOLVE_TIMEOUT_SECONDS) or notes._try_claude_api_format(prompt)
        if reply is None:
            reason = notes.last_cli_failure or "no reply"
            if not reason.startswith("the Claude CLI did not finish"):
                return {"ok": False, "claude_down": True, "message": f"Claude couldn't be reached ({reason})"}
            problem = f"Claude couldn't be reached ({reason})"
            continue
        reply = reply.strip()
        problem = resolve_problem(original, reply)
        if problem is None:
            break
    else:
        return {"ok": False, "message": f"Kept the notes as they were: {problem}"}
    current = path.read_text(encoding="utf-8")
    if current != content:
        unit_now = find_unit(current, key)
        if unit_now["text"].strip() != original:
            return {"ok": False, "message": "The notes changed while Claude was working; try again"}
        content, unit = current, unit_now
    lead = unit["text"][:len(unit["text"]) - len(unit["text"].lstrip())]
    trail = unit["text"][len(unit["text"].rstrip()):] or "\n"
    updated = content[:unit["start"]] + lead + reply + trail + content[unit["end"]:]
    _write(code, content, updated)
    left = _uncertain_lines(reply)
    return {"ok": True, "message": "Uncertain spots resolved" + (f" ({left} hedged line(s) kept)" if left else "")}


def move_session(payload: dict) -> dict:
    """Moves one recorded lecture to another class: its notes block goes into the other class's
    notes (in date order) and its recording files are renamed to the new class, so search,
    corrections and highlights all follow it. Refuses rather than overwrite anything."""
    code, session_id, target = (str(payload.get(k, "")) for k in ("class_code", "session_id", "target"))
    if target == code:
        raise ValueError("That lecture is already in this class")
    if target not in class_codes():
        raise ValueError("Pick a class that has notes")
    if session_class(session_id) != code:
        raise ValueError("That recording doesn't belong to this class")
    _refuse_while_recording()
    if current_job() and current_job()["status"] == "running":
        raise RepairBusy("A cleanup job is running; move lectures once it finishes")
    source_path, target_path = notes_file(code), notes_file(target)
    source = source_path.read_text(encoding="utf-8")
    unit = find_unit(source, "session:" + session_id)
    new_id = target.replace(" ", "_") + session_id[len(code.replace(" ", "_")):]
    target_content = target_path.read_text(encoding="utf-8")
    renames = []
    for old in STATE_DIR.glob(session_id + "*"):
        rest = old.name[len(session_id):]
        if rest[:1] not in (".", "_"):
            continue
        new = old.with_name(new_id + rest)
        if new.exists():
            raise RepairRefused(f"{new.name} already exists, so nothing was moved")
        renames.append((old, new))
    if session_markers(new_id)[0] in target_content:
        raise RepairRefused("That lecture is already in the other class's notes")

    body = unit["text"].strip()
    if not _DAY.search(body):
        body = f"## {unit['day']}\n\n{body}"  # it sat under an earlier recording's heading

    # Into the other class, in date order (session ids sort by time within a class).
    start_marker, end_marker = session_markers(new_id)
    block = f"{start_marker}\n{body}\n{end_marker}"
    stamp = new_id[-15:]
    later = [m for m in _SESSION_BLOCK.finditer(target_content) if m.group(1)[-15:] > stamp]
    if later:
        at = later[0].start()
        updated_target = target_content[:at] + block + "\n\n" + target_content[at:]
    else:
        updated_target = target_content.rstrip() + "\n\n" + block + "\n"

    # Out of this class. If it held the day's heading and the next recording of that day
    # relied on it, the heading moves down to that recording.
    left, right = unit["start"], unit["end"]
    block_start = source.rfind(session_markers(session_id)[0], 0, left)
    block_end = source.find(session_markers(session_id)[1], right) + len(session_markers(session_id)[1])
    remainder = source[block_end:]
    own_heading = _DAY.search(unit["text"])
    next_block = _SESSION_BLOCK.match(remainder.lstrip("\n"))
    same_day = next_block is not None and next_block.group(1)[-15:-7] == session_id[-15:-7]
    if own_heading and same_day and not _DAY.search(next_block.group(2).lstrip().split("\n", 1)[0]):
        updated_source = source[:block_start] + remainder.lstrip("\n")
        updated_source = update_session(updated_source, next_block.group(1),
                                        f"## {own_heading.group(1).strip()}\n\n{next_block.group(2).strip()}",
                                        replace=True)
    else:
        updated_source = source[:block_start].rstrip("\n") + "\n\n" + remainder.lstrip("\n")
    updated_source = re.sub(r"\n{3,}", "\n\n", updated_source)

    _write(target, target_content, updated_target)
    try:
        _write(code, source, updated_source)
    except Exception:
        atomic_write(target_path, target_content)  # put the other class back as it was
        raise
    for old, new in renames:
        os.replace(old, new)
        if new.name.endswith("_highlights.json"):
            try:
                data = json.loads(new.read_text(encoding="utf-8"))
                data["session"] = new_id
                atomic_write(new, json.dumps(data, ensure_ascii=False, indent=1), backup=False)
            except (OSError, ValueError):
                pass
    return {"ok": True, "message": f"Moved to {target}", "session_id": new_id}


# ---------------------------------------------------------------------------
# The background job
# ---------------------------------------------------------------------------

def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def _job_alive(job: dict) -> bool:
    if not job.get("pid"):
        return time.time() - job.get("created_at", 0) < 60  # just started, not reported in yet
    try:
        import psutil
        command = psutil.Process(job["pid"]).cmdline()
    except Exception:
        return False
    return any(part.endswith("repair.py") for part in command) and job.get("id", "") in " ".join(command)


def current_job() -> dict | None:
    job = _read_json(JOB_PATH, None)
    if not isinstance(job, dict):
        return None
    if job.get("status") == "running" and not _job_alive(job):
        job.update(status="failed", finished_at=time.time(),
                   message="The cleanup job stopped unexpectedly. Details: state/repair_job.log")
        for item in job.get("items", []):
            if item.get("state") in ("queued", "running"):
                item["state"] = "skipped"
        _write_json(JOB_PATH, job)
    return job


def plan_items(code: str, keys: list[str] | None = None, action: str = "fix") -> list[dict]:
    """The work a job does for one class: `keys` (all lectures that need anything when None)."""
    data = overview(code)
    items = []
    for row in data["units"]:
        if keys is not None and row["key"] not in keys:
            continue
        needs = {"reformat": row["has_transcript"],
                 "resolve": row["uncertain"] > 0,
                 "fix": (row["rough"] and row["has_transcript"]) or row["uncertain"] > 0}[action]
        if keys is None and not needs:
            continue
        label = row["day"] + (f" · {row['time']}" if row["time"] else "")
        items.append({"class_code": code, "key": row["key"], "label": label, "action": action,
                      "state": "queued", "message": ""})
    return items


def start_job(payload: dict, trigger: str = "dashboard") -> dict:
    """Starts a cleanup job in its own process. payload: {class_code, keys?: [...], action?}."""
    code = str(payload.get("class_code", ""))
    action = payload.get("action") or "fix"
    if action not in ("fix", "reformat", "resolve"):
        raise ValueError("Unknown cleanup action")
    keys = payload.get("keys")
    if keys is not None and (not isinstance(keys, list) or not all(isinstance(k, str) for k in keys)):
        raise ValueError("keys must be a list of lecture keys")
    _refuse_while_recording()
    items = plan_items(code, keys, action)
    if not items:
        return {"ok": True, "message": "Nothing to fix", "job": current_job()}
    return {"ok": True, "job": _spawn(items, trigger)}


def _spawn(items: list[dict], trigger: str) -> dict:
    with _job_lock:
        job = current_job()
        if job and job.get("status") == "running":
            raise RepairBusy("A cleanup job is already running; wait for it to finish")
        job = {"id": uuid.uuid4().hex, "status": "running", "trigger": trigger, "created_at": time.time(),
               "pid": None, "items": items, "message": ""}
        _write_json(JOB_PATH, job)
        log = open(STATE_DIR / "repair_job.log", "w", encoding="utf-8")
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        # The job process records its own pid when it starts (run_job); writing it from here
        # as well could land after the job's first progress update and wipe it out.
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--job", job["id"]],
                         cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        log.close()
    return job


def cancel_job(payload: dict | None = None) -> dict:
    job = current_job()
    if not job or job.get("status") != "running":
        return {"ok": True, "message": "No cleanup job is running"}
    job["cancel"] = True
    _write_json(JOB_PATH, job)
    return {"ok": True, "message": "The job stops after the lecture it's working on"}


def _memory_skip(memory: dict, item: dict, text: str) -> bool:
    entry = memory.get(item["class_code"] + "|" + item["key"])
    return bool(entry and entry.get("digest") == _digest(text)
                and time.time() - entry.get("at", 0) < FAILED_RETRY_SECONDS)


def _run_item(item: dict) -> dict:
    code, key = item["class_code"], item["key"]
    content = notes_file(code).read_text(encoding="utf-8")
    unit = find_unit(content, key)
    messages = []
    if item["action"] in ("fix", "reformat") and unit["kind"] == "session":
        if item["action"] == "reformat" or notes.HEURISTIC_MARKER in unit["text"]:
            result = reformat(code, unit["session_id"])
            if not result["ok"]:
                return result
            messages.append(result["message"])
    if item["action"] in ("fix", "resolve"):
        content = notes_file(code).read_text(encoding="utf-8")
        if _uncertain_lines(find_unit(content, key)["text"]):
            result = resolve(code, key)
            if not result["ok"]:
                result["message"] = "; ".join(messages + [result["message"]])
                return result
            messages.append(result["message"])
    return {"ok": True, "message": "; ".join(messages) or "Nothing needed fixing"}


def run_job(job_id: str) -> int:
    job = _read_json(JOB_PATH, {})
    if job.get("id") != job_id:
        print("That cleanup job was replaced by another one")
        return 1
    job["pid"] = os.getpid()
    _write_json(JOB_PATH, job)
    memory = _read_json(MEMORY_PATH, {})
    automatic = job.get("trigger") != "dashboard"
    stopped = None
    for index, item in enumerate(job["items"]):
        latest = _read_json(JOB_PATH, {})
        if latest.get("cancel"):
            stopped = "Cancelled"
        if stopped is None and recording_active():
            stopped = "A lecture started recording; the rest waits for the next cleanup"
        if stopped:
            for rest in job["items"][index:]:
                rest.update(state="skipped", message=stopped)
            break
        item["state"] = "running"
        item["started_at"] = time.time()
        _write_json(JOB_PATH, job)
        print(f"{item['class_code']} {item['label']}: {item['action']}...", flush=True)
        try:
            text = find_unit(notes_file(item["class_code"]).read_text(encoding="utf-8"), item["key"])["text"]
            if automatic and _memory_skip(memory, item, text):
                result = {"ok": False, "message": "Skipped: the last try failed and the notes haven't changed since"}
            else:
                result = _run_item(item)
        except Exception as error:
            result = {"ok": False, "message": f"{type(error).__name__}: {error}"}
        item["state"] = "done" if result["ok"] else "failed"
        item["message"] = result["message"]
        item["finished_at"] = time.time()
        print("  " + result["message"], flush=True)
        memory_key = item["class_code"] + "|" + item["key"]
        if result["ok"]:
            memory.pop(memory_key, None)
        elif not result.get("claude_down"):
            try:
                after = find_unit(notes_file(item["class_code"]).read_text(encoding="utf-8"), item["key"])["text"]
                memory[memory_key] = {"digest": _digest(after), "at": time.time(), "message": result["message"]}
            except Exception:
                pass
        _write_json(MEMORY_PATH, memory)
        if result.get("claude_down"):
            stopped = "Claude isn't reachable right now; the rest waits for the next cleanup"
        _write_json(JOB_PATH, job)
    job["status"] = "cancelled" if stopped == "Cancelled" else "finished"
    job["finished_at"] = time.time()
    done = sum(1 for i in job["items"] if i["state"] == "done")
    job["message"] = stopped or (f"{done} of {len(job['items'])} lecture(s) fixed" if job["items"]
                                 else "Nothing needed fixing")
    _write_json(JOB_PATH, job)
    if automatic:
        _housekeeping()
    return 0


def _housekeeping():
    """Compresses finished recordings and ages out very old ones (archive.py)."""
    try:
        import archive
        archive.compress_finished()
        archive.prune_old()
    except Exception as error:
        print(f"Recording housekeeping skipped ({type(error).__name__}: {error})")


def auto_items(lecture: tuple[str, str] | None = None) -> list[dict]:
    """What an automatic cleanup does: the lecture just recorded gets both fixes, and every
    lecture that's still roughly formatted (and has a transcript) is re-formatted."""
    items = []
    if lecture:
        code, session_id = lecture
        try:
            items += plan_items(code, ["session:" + session_id], "fix")
        except (FileNotFoundError, ValueError):
            pass
    for code in class_codes():
        try:
            rows = overview(code)["units"]
        except (FileNotFoundError, ValueError):
            continue
        for row in rows:
            if row["rough"] and row["has_transcript"] and not any(i["key"] == row["key"] and i["class_code"] == code
                                                                  for i in items):
                items.append({"class_code": code, "key": row["key"],
                              "label": row["day"] + (f" · {row['time']}" if row["time"] else ""),
                              "action": "reformat", "state": "queued", "message": ""})
    return items


def start_auto(lecture: tuple[str, str] | None = None) -> dict | None:
    """Starts an automatic cleanup unless one is running or a lecture is recording. With
    nothing to fix no job is started (the last job's results stay on the Cleanup page) and
    the caller does the recording housekeeping itself."""
    if recording_active():
        return None
    job = current_job()
    if job and job.get("status") == "running":
        return None
    items = auto_items(lecture)
    if not items:
        return None
    return _spawn(items, "after_lecture" if lecture else "auto")


def spawn_auto(lecture: tuple[str, str] | None = None):
    """Fire-and-forget from the recorder or dashboard: `repair.py --auto` in its own process,
    so working out what to fix never delays their exit or start-up."""
    args = ["--auto"] + (["--lecture", lecture[0], lecture[1]] if lecture else [])
    _spawn_detached(args)


def spawn_housekeeping():
    """Only the recording housekeeping, for lectures saved in a local-only formatting mode."""
    _spawn_detached(["--housekeeping"])


def _spawn_detached(extra: list[str]):
    args = [sys.executable, str(Path(__file__).resolve()), *extra]
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    try:
        subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags)
    except OSError:
        pass


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Repair lecture notes (rough formatting, uncertain spots)")
    parser.add_argument("--job", help="run the cleanup job with this id (started by start_job)")
    parser.add_argument("--auto", action="store_true", help="start an automatic cleanup")
    parser.add_argument("--lecture", nargs=2, metavar=("CLASS", "SESSION"), help="with --auto: the lecture just recorded")
    parser.add_argument("--housekeeping", action="store_true", help="only compress/age out recordings")
    args = parser.parse_args(argv)
    if args.job:
        return run_job(args.job)
    if args.housekeeping:
        if not recording_active():
            _housekeeping()
        return 0
    if args.auto:
        job = start_auto(tuple(args.lecture) if args.lecture else None)
        if job is None and not recording_active():
            _housekeeping()  # a started job does this itself when it finishes
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
