"""Notices a recording filed under the wrong class.

The recorder picks the class from the schedule, so an online lecture watched at another
class's time (or the wrong menu choice) lands in the wrong notes: a biology video about
hexose sugars ended up in WRTG 1310. Each class's notes make a vocabulary profile
(TF-IDF over the words in them), and a transcript that matches another class far better
than its own is flagged.

Tested on every saved transcript, comparing each against the notes with its own lecture
removed (that's the situation during a recording, before its notes exist). The thresholds
leave first-day syllabus talk alone (it matches whatever class has the most logistics), and
a class with almost no notes yet has no profile to judge by, so it's never flagged against.
"""
import collections
import math
import re
import threading
from pathlib import Path

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"

MIN_WORDS = 30              # shorter than this says nothing about the subject
LIVE_MIN_WORDS = 300        # during a recording, wait for a few minutes of lecture first
MIN_SCORE = 0.05            # the other class must actually match, not just match less badly
LIVE_MIN_SCORE = 0.10       # stricter mid-lecture: the first minutes are often just logistics
MIN_RATIO = 3.0             # ...and at least three times as well as the class it's filed under
MIN_PROFILE_WORDS = 1000    # a class needs this many content words (4+ letters) of notes to be judged

_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_SESSION_BLOCK = re.compile(r"<!-- session:([A-Za-z0-9_-]+) -->(.*?)<!-- /session:\1 -->", re.S)
_lock = threading.Lock()
_cache = {"key": None, "profiles": None}


def _words(text: str) -> list[str]:
    return _WORD.findall(re.sub(r"<!--.*?-->", " ", text).lower())


def _profiles() -> dict:
    """{code: {"total": Counter, "sessions": {session_id: Counter}}}, rebuilt when a notes file changes."""
    files = sorted(p for p in NOTES_DIR.glob("*.md") if not p.stem.endswith("_study_guide"))
    key = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    with _lock:
        if _cache["key"] == key:
            return _cache["profiles"]
    profiles = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        sessions = {sid: collections.Counter(_words(body)) for sid, body in _SESSION_BLOCK.findall(text)}
        profiles[path.stem.replace("_", " ")] = {"total": collections.Counter(_words(text)), "sessions": sessions}
    with _lock:
        _cache.update(key=key, profiles=profiles)
    return profiles


def _vector(counts: collections.Counter, df: collections.Counter, n: int) -> dict:
    # Words in every class's notes say nothing about which class this is.
    return {w: (1 + math.log(c)) * math.log((n + 1) / (df[w] + 1)) for w, c in counts.items() if c > 0 and df[w] < n}


def _cosine(a: dict, b: dict) -> float:
    dot = sum(v * b.get(w, 0.0) for w, v in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def scores(text: str, exclude_session: str | None = None) -> dict:
    """{class code: similarity} of a transcript to each class's notes (0..1)."""
    classes = {}
    for code, profile in _profiles().items():
        counts = profile["total"]
        if exclude_session and exclude_session in profile["sessions"]:
            counts = counts - profile["sessions"][exclude_session]
        classes[code] = counts
    if not classes:
        return {}
    df = collections.Counter()
    for counts in classes.values():
        df.update(w for w, c in counts.items() if c > 0)
    n = len(classes)
    query = _vector(collections.Counter(_words(text)), df, n)
    return {code: _cosine(query, _vector(counts, df, n)) for code, counts in classes.items()}


def suggest(text: str, current_code: str, exclude_session: str | None = None,
            min_words: int = MIN_WORDS, min_score: float = MIN_SCORE) -> dict | None:
    """{"code", "score", "current_score"} when the transcript clearly belongs to another
    class, else None."""
    if len(_words(text)) < min_words:
        return None
    profiles = _profiles()
    current = profiles.get(current_code)
    if current is None:
        return None
    own_words = sum(current["total"].values())
    if exclude_session and exclude_session in current["sessions"]:
        own_words -= sum(current["sessions"][exclude_session].values())
    if own_words < MIN_PROFILE_WORDS:
        return None
    ranked = scores(text, exclude_session)
    own = ranked.get(current_code, 0.0)
    best_code, best = max(((c, s) for c, s in ranked.items() if c != current_code),
                          key=lambda item: item[1], default=(None, 0.0))
    if best_code is None or best < min_score or best < MIN_RATIO * max(own, 0.01):
        return None
    return {"code": best_code, "score": round(best, 3), "current_score": round(own, 3)}
