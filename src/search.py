"""Full-text search across formatted notes and raw session transcripts.

Two different corpora, deliberately kept distinct in the results:
  - notes/<CODE>.md      - the polished, LLM-formatted study notes, split by lecture date
  - state/<CODE>_*_raw.txt - the raw timestamped transcript logs from each session

The notes are what you actually study from, so they're the primary hit. The raw logs
are the fallback for "I know it was said but it didn't make it into the notes", and
they carry a wall-clock timestamp, which is the closest thing this app has to
"jump to that moment in the recording" - the matching .wav sits next to the log.
"""
import re
import json
from transcripts import load_transcript
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = PROJECT_ROOT / "notes"
STATE_DIR = PROJECT_ROOT / "state"

_DATE_HEADING = re.compile(r"^## (.+)$")
_RAW_LINE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s?(.*)$")
_BACKUP_NAME = re.compile(r"^(.+)_(\d{8})_(\d{6})_raw\.txt$")


@dataclass
class Hit:
    class_code: str
    source: str          # "notes" or "transcript"
    location: str        # lecture date heading, or session timestamp
    text: str            # the matching line
    line_number: int
    timestamp: str = ""  # wall-clock time, transcript hits only
    audio_file: str = ""
    audio_seconds: float | None = None
    section: str = ""    # nearest "###" subheading, notes hits only
    context: list = field(default_factory=list)  # neighbouring lines, for display


def _class_codes() -> list[str]:
    if not NOTES_DIR.exists():
        return []
    return sorted(p.stem.replace("_", " ") for p in NOTES_DIR.glob("*.md")
                  if not p.stem.endswith("_study_guide"))


def _matches(line: str, terms: list[str], match_all: bool) -> bool:
    lowered = line.lower()
    if match_all:
        return all(t in lowered for t in terms)
    return any(t in lowered for t in terms)


def search_notes(query: str, class_code: str | None = None, match_all: bool = True,
                  context_lines: int = 1) -> list[Hit]:
    """Searches the formatted per-lecture notes. Tracks the current '## <date>' and
    '### <topic>' headings while scanning so each hit can say which lecture and which
    part of it the line came from, rather than just a bare line number."""
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []

    hits = []
    codes = [class_code] if class_code in _class_codes() else ([] if class_code else _class_codes())
    for code in codes:
        path = NOTES_DIR / f"{code.replace(' ', '_')}.md"
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        current_date, current_section = "", ""
        for i, line in enumerate(lines):
            heading = _DATE_HEADING.match(line)
            if heading:
                current_date, current_section = heading.group(1).strip(), ""
                continue
            if line.startswith("### "):
                current_section = line[4:].strip()
            if line.startswith("<!--"):
                continue
            if not _matches(line, terms, match_all) or not line.strip():
                continue
            if line.startswith("#"):
                continue  # headings themselves are navigation, not content
            hits.append(Hit(
                class_code=code, source="notes", location=current_date or "(undated)",
                text=line.strip(), line_number=i + 1, section=current_section,
                context=[l.strip() for l in lines[max(0, i - context_lines): i + context_lines + 1]],
            ))
    return hits


def search_transcripts(query: str, class_code: str | None = None,
                        match_all: bool = True) -> list[Hit]:
    """Searches the raw timestamped session logs in state/. Slower and noisier than the
    notes (this is unedited speech-to-text) but it's the only place something the notes
    dropped can still be found, and every hit carries the timestamp it was said at."""
    terms = [t for t in query.lower().split() if t]
    if not terms or not STATE_DIR.exists():
        return []

    hits = []
    for path in sorted(STATE_DIR.glob("*_raw.txt")):
        name_match = _BACKUP_NAME.match(path.name)
        if not name_match:
            continue
        code = name_match.group(1).replace("_", " ")
        if class_code and code != class_code:
            continue
        session = f"{name_match.group(2)} {name_match.group(3)}"
        wav = path.with_name(path.name.removesuffix("_raw.txt") + ".wav")
        sidecar = path.with_name(path.name.removesuffix("_raw.txt") + "_segments.jsonl")
        overlay = wav.with_name(wav.stem + "_corrections.json")
        if overlay.exists():
            corrected = load_transcript(STATE_DIR, wav.stem)
            for segment in corrected["segments"]:
                if not _matches(segment["text"], terms, match_all):
                    continue
                seconds = segment["start"]
                hits.append(Hit(class_code=code, source="transcript", location=session,
                                text=segment["text"], line_number=segment["index"]+1,
                                timestamp=(f"{int(seconds)//60:02}:{int(seconds)%60:02}" if seconds is not None else ""),
                                audio_file=wav.name if wav.exists() else "",audio_seconds=seconds))
            continue
        if sidecar.exists():
            # Audio-relative offsets are independent of transcription latency.
            for i, line in enumerate(sidecar.read_text(encoding="utf-8").splitlines()):
                try:
                    segment = json.loads(line)
                    text = str(segment["text"])
                    seconds = float(segment["start"])
                except (ValueError, KeyError, TypeError):
                    continue  # an interrupted final append must not hide earlier segments
                if _matches(text, terms, match_all):
                    hits.append(Hit(class_code=code, source="transcript", location=session,
                                    text=text, line_number=i + 1,
                                    timestamp=f"{int(seconds)//60:02}:{int(seconds)%60:02}",
                                    audio_file=wav.name if wav.exists() else "",
                                    audio_seconds=max(0, seconds)))
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if not _matches(line, terms, match_all):
                continue
            stamped = _RAW_LINE.match(line)
            hits.append(Hit(
                class_code=code, source="transcript", location=session,
                text=(stamped.group(2) if stamped else line).strip(),
                line_number=i + 1,
                timestamp=stamped.group(1) if stamped else "",
                audio_file=wav.name if wav.exists() else "",
            ))
    return hits


def search(query: str, class_code: str | None = None, include_transcripts: bool = True,
            match_all: bool = True) -> list[Hit]:
    hits = search_notes(query, class_code, match_all)
    if include_transcripts:
        hits += search_transcripts(query, class_code, match_all)
    return hits
