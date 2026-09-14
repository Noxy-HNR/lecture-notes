"""Shared transcript reading and reversible correction overlays."""
import hashlib
import json
import math
import re
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def session_class(session_id):
    match = re.fullmatch(r"([A-Za-z0-9_-]+)_(\d{8})_(\d{6})", session_id)
    if not match:
        raise ValueError("Invalid recording ID")
    return match[1].replace("_", " ")


def session_path(directory, session_id, suffix):
    session_class(session_id)
    directory = Path(directory).resolve()
    path = (directory / (session_id + suffix)).resolve()
    if path.parent != directory:
        raise ValueError("Recording path is outside the library")
    return path


def load_transcript(directory, session_id):
    source = session_path(directory, session_id, "_segments.jsonl")
    timed = source.exists()
    if not timed:
        source = session_path(directory, session_id, "_raw.txt")
    if not source.is_file():
        raise FileNotFoundError("This recording has no transcript yet")
    text = source.read_text(encoding="utf-8")
    segments = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if timed:
            try:
                entry = json.loads(line)
                start, end = float(entry["start"]), float(entry.get("end", entry["start"]))
                if not all(math.isfinite(v) for v in (start,end)) or end < start:
                    continue
                words = str(entry["text"])
            except (ValueError,KeyError,TypeError):
                continue  # keep complete segments after an interrupted final append
        else:
            words = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s?", "", line)
            start = end = None  # legacy wall time is not an audio offset
        segments.append({"index":len(segments), "start":start, "end":end,
                         "original":words, "text":words})
    overlay_path = session_path(directory,session_id,"_corrections.json")
    overlay_text = overlay_path.read_text(encoding="utf-8") if overlay_path.exists() else ""
    overlay = json.loads(overlay_text) if overlay_text else {}
    source_revision = digest(text)
    if overlay and overlay.get("source_revision") != source_revision:
        raise ValueError("The source transcript changed after it was corrected. Restore its matching revision before editing or recovering it.")
    edits = overlay.get("edits", {})
    for segment in segments:
        segment["text"] = edits.get(str(segment["index"]),segment["original"])
    return {"segments":segments, "source_revision":source_revision,
            "revision":digest(text + "\0" + overlay_text), "overlay_text":overlay_text,
            "overlay_path":overlay_path, "edits":edits, "timed":timed}
