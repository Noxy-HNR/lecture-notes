"""Lecture highlights: what each recording says about the exam, and where it was probably mis-heard.

Jev (jev.py) reads each transcript segment - about 30 seconds of speech - and answers four
yes/no questions about it in one call:

  exam          the instructor marks something specific as testable or required
  not_tested    the instructor says a specific detail won't be tested / needn't be memorized
  announcement  a due date, exam date, assignment, grading or schedule change, office hours...
  mishearing    a word is probably a speech-recognition error for a term from the course

The Notes page shows the first three above each lecture's notes, with the moment in the
recording; the Corrections page lists the fourth as "check these". The notes files are not
changed.

Results live beside the recording as state/<recording>_highlights.json, holding probabilities
and a digest of what was asked per segment - no transcript text. Keeping the raw probabilities
means the thresholds below can change without asking Jev again, and a segment is only re-asked
when its text (or its neighbours', or the questions) changes, e.g. after a correction.

Scanning runs in a background thread of the dashboard process (Scanner), newest recording
first, and never touches a recording that is still in progress.
"""
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import jev
from transcripts import load_transcript, session_class, session_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "state"

# Probability at or above which a segment is listed, chosen by reading every judgment on three
# real lectures (238 segments; README, "Lecture highlights"). Exam hints sit lower because
# missing one costs more than reading an extra line. Speech recognition errs often, so about a
# third of segments score over 0.5 for a mis-hearing - mostly real errors, but too many to
# check; 0.7 keeps the clearest. The stored probabilities make changing these free.
THRESHOLDS = {"exam": 0.4, "not_tested": 0.5, "announcement": 0.5, "mishearing": 0.7}

PASSAGE_CHARS = 1600   # a segment is ~30 s of speech, usually 400-700 characters
BEFORE_CHARS = 300     # enough to finish a sentence cut at the segment's start
AFTER_CHARS = 200
TERMS_LIMIT = 80

_CONTEXT = ("`passage` is part of an automatic transcript of a college lecture in `course`. "
            "`before` and `after` are the words spoken just before and after it, given only so that "
            "sentences cut off at the edges of `passage` make sense; judge `passage` itself. ")

QUESTIONS = {
    "exam": {
        "type": "noul",
        "instructions": _CONTEXT + (
            "In `passage`, does the instructor tell students that something specific will be on an "
            "exam, quiz or test, or that they need to know, remember or be able to do something "
            "specific for this course?"),
        "criteria": {
            "true": ("Within `passage`, the instructor marks specific material as testable or required, "
                     "for example: 'this will be on the exam', 'you need to know these three types', "
                     "'make sure you can explain this', 'I want you to remember this', "
                     "'I may ask you a question like this', 'you should be able to draw this', "
                     "'this is a common test question'."),
            "false": ("No such statement within `passage`. Ordinary teaching, examples, demonstrations, "
                      "general study advice, talk about grades or past exam scores, and statements that "
                      "something does not need to be memorized all count as no."),
        },
    },
    "not_tested": {
        "type": "noul",
        "instructions": _CONTEXT + (
            "In `passage`, does the instructor tell students that a specific detail will not be tested, "
            "or that they do not need to memorize or know it?"),
        "criteria": {
            "true": ("Within `passage`, the instructor says a specific detail is not required, for example: "
                     "'you don't need to memorize this number', 'this won't be on the test', "
                     "'just know the general idea, not the dates'."),
            "false": "No such statement within `passage`.",
        },
    },
    "announcement": {
        "type": "noul",
        "instructions": _CONTEXT + (
            "In `passage`, does the instructor announce course logistics that a student would want to "
            "put on a to-do list or calendar?"),
        "criteria": {
            "true": ("Within `passage`, the instructor gives a due date, an exam or quiz date, an assignment "
                     "or reading to do, a change to grading or to the schedule, extra credit, office hours "
                     "or another way to get help, or something to bring or submit."),
            "false": ("No such announcement within `passage`. Teaching content, examples, questions to the "
                      "class, and activities happening in the room right now (such as handing out a sheet) "
                      "count as no."),
        },
    },
    "mishearing": {
        "type": "noul",
        "instructions": (
            "`passage` was written by speech recognition from a college lecture in `course`. Speech "
            "recognition sometimes writes a similar-sounding wrong word, or misspells a name or technical "
            "term. `course_terms`, when present, lists some terms this course uses. Does `passage` "
            "contain at least one word or name that is probably such an error, where a similar-sounding "
            "word, name or term that fits the lecture is what the instructor most likely said?"),
        "criteria": {
            "true": ("At least one word or short phrase in `passage` makes little sense where it is, and a "
                     "similar-sounding word or term fits the lecture, for example 'sell membrane' for "
                     "'cell membrane', 'the super eagle' for 'the superego', or a scientist's name spelled "
                     "as a different name."),
            "false": ("Every word is plausibly what was said. Casual speech, filler words, repeated words, "
                      "sentences cut off at the edges of `passage`, and small grammar slips are not "
                      "speech-recognition errors."),
        },
    },
}
KINDS = tuple(QUESTIONS)
# Part of every stored digest: editing a question re-asks it everywhere, nothing else does.
QUESTIONS_VERSION = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()[:12]
FORMAT_VERSION = 1

_STAMP = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]")


# ---------------------------------------------------------------------------
# What is asked
# ---------------------------------------------------------------------------

def course_label(code: str) -> str:
    try:
        import schedule
        found = schedule.get_class_by_code(code)
        title = (found or {}).get("title")
    except Exception:
        title = None
    return f"{title} ({code})" if title else code


def course_terms(code: str) -> list[str]:
    try:
        import vocab
        terms = vocab.load_vocab().get(code, [])
    except Exception:
        return []
    return [str(t) for t in terms if isinstance(t, str)][:TERMS_LIMIT]


def segment_states(code: str, segments: list[dict]) -> list[dict]:
    """The state Jev sees for each segment, already redacted and clipped."""
    course = course_label(code)
    terms = course_terms(code)
    texts = [jev.redact(s["text"]) for s in segments]
    states = []
    for i, text in enumerate(texts):
        state = {"course": course,
                 "before": jev.clip(texts[i - 1], BEFORE_CHARS, keep="end") if i else "",
                 "passage": jev.clip(text, PASSAGE_CHARS),
                 "after": jev.clip(texts[i + 1], AFTER_CHARS) if i + 1 < len(texts) else ""}
        if terms:
            state["course_terms"] = terms
        states.append(state)
    return states


def _digest(state: dict) -> str:
    # The glossary is left out: it only helps, and a term learned from one correction
    # shouldn't re-ask every segment the class has ever recorded.
    asked = {k: v for k, v in state.items() if k != "course_terms"}
    raw = QUESTIONS_VERSION + json.dumps(asked, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Stored results
# ---------------------------------------------------------------------------

def result_path(session_id: str) -> Path:
    return session_path(STATE_DIR, session_id, "_highlights.json")


def read_result(session_id: str) -> dict | None:
    try:
        data = json.loads(result_path(session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == FORMAT_VERSION else None


def _write_result(session_id: str, data: dict):
    path = result_path(session_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def transcript_sessions() -> list[str]:
    """Every recording with a transcript, newest first."""
    ids = {p.name.removesuffix("_raw.txt") for p in STATE_DIR.glob("*_raw.txt")}
    ids.update(p.name.removesuffix("_segments.jsonl") for p in STATE_DIR.glob("*_segments.jsonl"))
    valid = []
    for sid in ids:
        try:
            session_class(sid)
        except ValueError:
            continue
        valid.append(sid)
    return sorted(valid, key=lambda s: s[-15:], reverse=True)


def live_session() -> str | None:
    """The recording in progress, if any: its transcript is still growing, so it waits."""
    try:
        import telemetry
        status = telemetry.read_status()
    except Exception:
        return None
    if not status.get("active") or time.time() - float(status.get("updated_at") or 0) > 120:
        return None
    wav = status.get("wav_path") or ""
    return Path(wav).stem or None


def needs_scan(session_id: str) -> bool:
    """True when some segment has never been asked, or was asked about different text."""
    try:
        transcript = load_transcript(STATE_DIR, session_id)
    except (FileNotFoundError, ValueError):
        return False
    states = segment_states(session_class(session_id), transcript["segments"])
    stored = read_result(session_id) or {}
    rows = stored.get("segments") or []
    if len(rows) != len(states):
        return True
    return any(row.get("digest") != _digest(state) or any(k not in row for k in KINDS)
               for row, state in zip(rows, states))


def scan_session(session_id: str, should_stop=None, progress=None) -> dict:
    """Asks Jev about every segment whose question text changed since the last scan (all of
    them, the first time) and saves the probabilities. Returns the stored result."""
    code = session_class(session_id)
    transcript = load_transcript(STATE_DIR, session_id)
    states = segment_states(code, transcript["segments"])
    previous = {row.get("digest"): row for row in (read_result(session_id) or {}).get("segments", [])
                if isinstance(row, dict) and all(k in row for k in KINDS)}
    rows = [None] * len(states)
    todo = []
    for i, state in enumerate(states):
        digest = _digest(state)
        if digest in previous:
            rows[i] = dict(previous[digest], index=i)
        else:
            todo.append((i, digest, state))
    usage = jev.Usage()
    errors = []
    if progress:
        progress(len(rows) - len(todo), len(rows))
    for start in range(0, len(todo), 40):  # batches, so progress and stop requests land promptly
        batch = todo[start:start + 40]
        answers = jev.ask_many([state for _, _, state in batch], QUESTIONS, usage=usage,
                               should_stop=should_stop)
        for (i, digest, _), answer in zip(batch, answers):
            if isinstance(answer, jev.JevError):
                errors.append(answer)
                continue
            try:
                rows[i] = {"index": i, "digest": digest,
                           **{kind: round(jev.noul(answer, kind), 4) for kind in KINDS}}
            except jev.JevError as error:
                errors.append(error)
        if progress:
            progress(sum(r is not None for r in rows), len(rows))
        if errors and errors[-1].kind in jev.FATAL_KINDS or (should_stop and should_stop()):
            break
    old = read_result(session_id) or {}
    spent = old.get("usage") or {}
    now = usage.as_dict()
    result = {
        "format": FORMAT_VERSION, "session": session_id, "questions": QUESTIONS_VERSION,
        "model": jev.MODEL, "scanned_at": time.time(),
        "complete": all(r is not None for r in rows),
        "segments": [r for r in rows if r is not None],
        "usage": {"calls": int(spent.get("calls", 0)) + now["calls"],
                  "input_tokens": int(spent.get("input_tokens", 0)) + now["input_tokens"],
                  "est_cost_usd": round(float(spent.get("est_cost_usd", 0)) + now["est_cost_usd"], 6)},
    }
    if errors:
        result["last_error"] = {"kind": errors[-1].kind, "message": str(errors[-1])}
    _write_result(session_id, result)
    return result


# ---------------------------------------------------------------------------
# What the pages show
# ---------------------------------------------------------------------------

def _clock(seconds) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02}:{rest % 60:02}" if hours else f"{rest // 60}:{rest % 60:02}"


def _wall_stamps(session_id: str) -> list[str]:
    """Legacy transcripts have no audio offsets; their raw lines carry the wall-clock time."""
    try:
        text = session_path(STATE_DIR, session_id, "_raw.txt").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    return [(m.group(1) if (m := _STAMP.match(line)) else "") for line in text.splitlines() if line.strip()]


def session_items(session_id: str, kinds=("exam", "not_tested", "announcement")) -> dict:
    """The segments over threshold for each kind, with their text and moment."""
    stored = read_result(session_id)
    out = {kind: [] for kind in kinds}
    if not stored:
        return out
    try:
        transcript = load_transcript(STATE_DIR, session_id)
    except (FileNotFoundError, ValueError):
        return out
    segments = transcript["segments"]
    states = segment_states(session_class(session_id), segments)
    stamps = None if transcript["timed"] else _wall_stamps(session_id)
    audio = session_id + ".wav" if session_path(STATE_DIR, session_id, ".wav").exists() else ""
    for row in stored.get("segments", []):
        i = row.get("index")
        if not isinstance(i, int) or not 0 <= i < len(segments) or row.get("digest") != _digest(states[i]):
            continue  # asked about text that has since changed; a rescan will replace it
        segment = segments[i]
        for kind in kinds:
            p = row.get(kind)
            if isinstance(p, (int, float)) and p >= THRESHOLDS[kind]:
                start = segment["start"]
                out[kind].append({
                    "session": session_id, "index": i, "p": p, "text": segment["text"],
                    "start": start, "audio_file": audio,
                    "clock": _clock(start) if start is not None else (stamps[i] if stamps and i < len(stamps) else ""),
                })
    return out


def _heading_date(text: str):
    try:
        return datetime.strptime(text.strip(), "%A, %B %d, %Y").date()
    except ValueError:
        return None


def for_class(code: str, headings: list[tuple[str, str]], scanner=None) -> dict:
    """Highlights per lecture heading. `headings` is [(date heading, anchor)] in document order;
    each recording goes under the first heading with its date, so a day recorded in several
    parts (or saved in several sections) gets one box."""
    first_anchor = {}
    for text, anchor in headings:
        day = _heading_date(text)
        if day and day not in first_anchor:
            first_anchor[day] = anchor
    live = live_session()
    lectures: dict[str, dict] = {}
    for sid in transcript_sessions():
        if session_class(sid) != code:
            continue
        try:
            day = datetime.strptime(sid[-15:-7], "%Y%m%d").date()
        except ValueError:
            continue
        anchor = first_anchor.get(day)
        if not anchor:
            continue
        box = lectures.setdefault(anchor, {"anchor": anchor, "sessions": [], "pending": [],
                                           "exam": [], "not_tested": [], "announcement": []})
        box["sessions"].append(sid)
        stored = read_result(sid)
        if sid == live:
            box["pending"].append("recording")
        elif not stored or not stored.get("complete"):
            box["pending"].append("scanning" if scanner and scanner.current == sid else "waiting")
        items = session_items(sid)
        for kind in ("exam", "not_tested", "announcement"):
            box[kind].extend(items[kind])
    for box in lectures.values():
        box["sessions"].sort()
        for kind in ("exam", "not_tested", "announcement"):
            box[kind].sort(key=lambda item: (item["session"], item["index"]))
            if len(box["sessions"]) > 1:  # each part's clock starts at 0:00, so say which part
                for item in box[kind]:
                    item["part"] = box["sessions"].index(item["session"]) + 1
    return {"jev": jev.status(), "lectures": list(lectures.values()),
            "scanner": scanner.snapshot() if scanner else None}


def mishearings(session_id: str) -> dict:
    """For the Corrections page: segment indices worth checking, most likely first."""
    stored = read_result(session_id)
    flagged = session_items(session_id, kinds=("mishearing",))["mishearing"]
    return {"jev": jev.status(), "scanned": bool(stored),
            "complete": bool(stored and stored.get("complete")),
            "threshold": THRESHOLDS["mishearing"],
            "segments": [{"index": item["index"], "p": item["p"]}
                         for item in sorted(flagged, key=lambda item: -item["p"])]}


# ---------------------------------------------------------------------------
# Background scanning
# ---------------------------------------------------------------------------

class Scanner(threading.Thread):
    """Scans recordings that need it, newest first, one at a time. Idle checks are cheap (a
    digest comparison per segment), so it looks again every minute; after a failure it waits
    longer, and a rejected key waits until the key changes."""

    IDLE_SECONDS = 60
    RETRY_SECONDS = 600

    def __init__(self):
        super().__init__(daemon=True, name="highlights-scanner")
        self.current = None
        self.progress = (0, 0)
        self.last_error = None
        self.last_scan = None
        self._halt = threading.Event()
        self._wake = threading.Event()
        self._retry_at = {}
        self._blocked_key = None

    def stop(self):
        self._halt.set()
        self._wake.set()

    def wake(self):
        self._wake.set()

    def snapshot(self) -> dict:
        done, total = self.progress
        return {"current": self.current, "done": done, "total": total,
                "last_error": self.last_error, "last_scan": self.last_scan}

    def _key_fingerprint(self):
        key = jev.api_key()[0]
        return hashlib.sha256(key.encode()).hexdigest() if key else None

    def pending(self) -> list[str]:
        live = live_session()
        now = time.time()
        return [sid for sid in transcript_sessions()
                if sid != live and self._retry_at.get(sid, 0) <= now and needs_scan(sid)]

    def run_once(self) -> bool:
        """Scans the next recording that needs it. Returns whether there was one."""
        if not jev.status()["ready"]:
            return False
        fingerprint = self._key_fingerprint()
        if self._blocked_key and fingerprint == self._blocked_key:
            return False
        self._blocked_key = None
        queue = self.pending()
        if not queue:
            return False
        sid = queue[0]
        self.current, self.progress = sid, (0, 0)
        try:
            result = scan_session(sid, should_stop=self._halt.is_set,
                                  progress=lambda done, total: setattr(self, "progress", (done, total)))
        except (FileNotFoundError, ValueError, OSError) as error:
            self.last_error = f"{sid}: {type(error).__name__}"
            self._retry_at[sid] = time.time() + self.RETRY_SECONDS
            return True
        finally:
            self.current = None
        self.last_scan = {"session": sid, "at": time.time(), "usage": result.get("usage")}
        error = result.get("last_error")
        if error:
            self.last_error = error["message"]
            if error["kind"] == "auth":
                self._blocked_key = fingerprint
            if not result.get("complete"):
                self._retry_at[sid] = time.time() + self.RETRY_SECONDS
        else:
            self.last_error = None
        return True

    def run(self):
        self._halt.wait(5)  # let the dashboard finish starting first
        while not self._halt.is_set():
            try:
                busy = self.run_once()
            except Exception as error:  # never let a scan take the dashboard down
                self.last_error = f"Scanner error: {type(error).__name__}"
                busy = False
            if not busy:
                self._wake.wait(self.IDLE_SECONDS)
                self._wake.clear()
