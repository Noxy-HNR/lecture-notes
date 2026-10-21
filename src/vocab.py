"""Per-class glossary of domain-specific and Latin-derived terms, used by the local
formatter to fix near-miss spellings in transcripts. Edit vocab.json to add your own
terms per class code (or under "_global" for terms that apply everywhere).
"""
import json
import sys
from pathlib import Path
from storage import atomic_write, RevisionConflict

VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.json"
_warned = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"Warning: {message}", file=sys.stderr)


def load_vocab() -> dict:
    """The glossary, or {} when vocab.json is missing or unusable. A typo in the file must never
    stop a recording or a notes save: it is reported once, and the file is left for you to fix."""
    try:
        if not VOCAB_PATH.exists():
            return {}
        # utf-8-sig: Windows editors and PowerShell often save the file with a byte-order mark.
        data = json.loads(VOCAB_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        _warn_once(f"vocab.json could not be read ({error}). The glossary is off until it is fixed.")
        return {}
    if not isinstance(data, dict):
        _warn_once("vocab.json should map class codes to lists of terms. The glossary is off until it is fixed.")
        return {}
    return data


def _term_list(value) -> list[str]:
    return [t for t in value if isinstance(t, str) and t.strip()] if isinstance(value, list) else []


def terms_for_class(class_code: str) -> list[str]:
    vocab = load_vocab()
    return _term_list(vocab.get("_global")) + _term_list(vocab.get(class_code))


def class_only_terms(class_code: str) -> list[str]:
    """This class's own terms, without the "_global" ones."""
    return _term_list(load_vocab().get(class_code))


def save_learned_terms(class_code: str, terms: set[str]) -> None:
    """Persists newly-learned terms into vocab.json under `class_code`, skipping any
    that are already present (case-insensitively) for that class or globally. Used to
    auto-grow the glossary from corrections the CLI/API proofreading pass makes, so the
    heuristic/local formatters catch the same terms next call without needing an LLM."""
    if not terms:
        return
    for attempt in range(3):
        before = VOCAB_PATH.read_text(encoding="utf-8") if VOCAB_PATH.exists() else ""
        data = json.loads(before.lstrip("﻿")) if before.strip() else {}  # the rewrite drops any BOM
        existing_lower = {t.lower() for t in data.get("_global", [])} | {t.lower() for t in data.get(class_code, [])}
        new_terms = sorted(t for t in terms if t.lower() not in existing_lower)
        if not new_terms:
            return
        data.setdefault(class_code, [])
        data[class_code].extend(new_terms)
        try:
            atomic_write(VOCAB_PATH,json.dumps(data,indent=2)+"\n",expected=before)
            return
        except RevisionConflict:
            if attempt == 2:
                raise
