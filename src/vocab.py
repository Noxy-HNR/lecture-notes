"""Per-class glossary of domain-specific and Latin-derived terms, used by the local
formatter to fix near-miss spellings in transcripts. Edit vocab.json to add your own
terms per class code (or under "_global" for terms that apply everywhere).
"""
import json
from pathlib import Path
from storage import atomic_write, RevisionConflict

VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.json"


def load_vocab() -> dict:
    if VOCAB_PATH.exists():
        return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return {}


def terms_for_class(class_code: str) -> list[str]:
    vocab = load_vocab()
    return vocab.get("_global", []) + vocab.get(class_code, [])


def save_learned_terms(class_code: str, terms: set[str]) -> None:
    """Persists newly-learned terms into vocab.json under `class_code`, skipping any
    that are already present (case-insensitively) for that class or globally. Used to
    auto-grow the glossary from corrections the CLI/API proofreading pass makes, so the
    heuristic/local formatters catch the same terms next call without needing an LLM."""
    if not terms:
        return
    for attempt in range(3):
        before = VOCAB_PATH.read_text(encoding="utf-8") if VOCAB_PATH.exists() else ""
        data = json.loads(before) if before else {}
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
