"""Figures out which class you're in right now (or about to be in) from schedule.json."""
import json
from datetime import datetime, timedelta
from pathlib import Path

SCHEDULE_PATH = Path(__file__).resolve().parent.parent / "schedule.json"


def load_schedule():
    with open(SCHEDULE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_hm(s):
    h, m = s.split(":")
    return int(h), int(m)


def _session_window(now, session):
    """Return (start_dt, end_dt) for a session on the date of `now`."""
    sh, sm = _parse_hm(session["start"])
    eh, em = _parse_hm(session["end"])
    start_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start_dt, end_dt


def all_sessions_today(now=None):
    """All (class, session) pairs scheduled on today's weekday, sorted by start time."""
    now = now or datetime.now()
    day_name = now.strftime("%A")
    data = load_schedule()
    results = []
    for cls in data["classes"]:
        for session in cls["sessions"]:
            if session["day"] == day_name:
                start_dt, end_dt = _session_window(now, session)
                results.append((cls, session, start_dt, end_dt))
    results.sort(key=lambda r: r[2])
    return results


def find_current_class(now=None, grace_minutes=10):
    """
    Returns a dict describing the class to transcribe right now, or None.

    Priority:
      1. A class currently in session (now between start and end).
      2. A class starting within `grace_minutes` (so you can start the app a
         little early and it still picks the right one).
      3. A class that ended within `grace_minutes` ago (you started the app
         a bit late but it's still worth capturing the tail end).
    """
    now = now or datetime.now()
    sessions = all_sessions_today(now)
    if not sessions:
        return None

    grace = timedelta(minutes=grace_minutes)

    # 1. Currently in session
    for cls, session, start_dt, end_dt in sessions:
        if start_dt <= now <= end_dt:
            return _make_result(cls, session, start_dt, end_dt, "in_session")

    # 2. Starting soon
    for cls, session, start_dt, end_dt in sessions:
        if now < start_dt <= now + grace:
            return _make_result(cls, session, start_dt, end_dt, "starting_soon")

    # 3. Just ended
    for cls, session, start_dt, end_dt in sessions:
        if now - grace <= end_dt < now:
            return _make_result(cls, session, start_dt, end_dt, "just_ended")

    return None


def _make_result(cls, session, start_dt, end_dt, status):
    return {
        "code": cls["code"],
        "title": cls["title"],
        "location": session["location"],
        "type": session["type"],
        "day": session["day"],
        "start": start_dt,
        "end": end_dt,
        "status": status,
    }


def list_all_classes():
    """Flat list of every class code+title, for manual selection."""
    data = load_schedule()
    return [(cls["code"], cls["title"]) for cls in data["classes"]]


def get_class_by_code(code):
    data = load_schedule()
    for cls in data["classes"]:
        if cls["code"].lower() == code.lower():
            return cls
    return None
