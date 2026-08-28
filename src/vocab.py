"""Per-class vocabulary hints that get fed to Whisper as an `initial_prompt`, which
measurably improves recognition of domain-specific and Latin-derived terms it
wouldn't otherwise guess correctly from audio alone. Edit vocab.json to add your
own terms per class code (or under "_global" for terms that apply everywhere).
"""
import json
from pathlib import Path

VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.json"


def load_vocab() -> dict:
    if VOCAB_PATH.exists():
        return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return {}


def terms_for_class(class_code: str) -> list[str]:
    vocab = load_vocab()
    return vocab.get("_global", []) + vocab.get(class_code, [])


def initial_prompt_for_class(class_code: str) -> str | None:
    """Whisper's initial_prompt is just a short text snippet used to bias recognition
    toward this vocabulary/style - it isn't a strict allowlist, just a nudge."""
    terms = terms_for_class(class_code)
    if not terms:
        return None
    return "Vocabulary that may appear in this lecture: " + ", ".join(terms) + "."
