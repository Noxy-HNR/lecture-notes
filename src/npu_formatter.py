"""Local, offline note formatting using a small LLM (Qwen2.5-1.5B-Instruct) running on
the machine's NPU (Intel AI Boost) via OpenVINO GenAI. Sits between the Claude CLI/API
and the pure heuristic formatter in notes.py's fallback chain: much more capable than
regex-based formatting (actual rewriting/cleanup, not just bullet-splitting) while
staying fully local, fast, and off the GPU that's busy transcribing.

Optional setup (see README "NPU-accelerated local formatting"):
  1. pip install openvino openvino-tokenizers openvino-genai
  2. Download the model once:
       from huggingface_hub import snapshot_download
       snapshot_download("OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov", local_dir="state/npu_model_phi35mini_gq")

If the packages aren't installed or the model isn't downloaded, format_transcript()
returns None and notes.py falls back further to the pure heuristic formatter - nothing
else breaks.

Model notes: Phi-3.5-mini-instruct (3.8B, group-quantized specifically for NPU) is used
over the smaller Qwen2.5-1.5B for better formatting quality. Two things tried and
rejected first: Qwen2.5-1.5B-Instruct-int4-ov (works, ~49s load/~5s generation, but
noticeably weaker output) and Phi-4-mini-instruct-int4-ov (crashed the NPU driver with
ZE_RESULT_ERROR_DEVICE_LOST on every generation attempt, reproducibly - do not use).
"""
import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "state" / "npu_model_phi35mini_gq"
MAX_NEW_TOKENS = 800

_pipeline = None
_load_failed = False

# Phi-3.5-mini doesn't reliably follow "no commentary" instructions - it sometimes
# appends process notes about its own corrections. Stripped as a safety net rather
# than relied on the prompt alone.
_META_COMMENTARY_PATTERNS = [
    re.compile(r"\n?\(Note:.*?\)\s*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n?#{0,3}\s*Speech-to-text Errors?:.*?(?=\n#{1,3}\s|\Z)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^.*\bas per the instructions?\b.*$\n?", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^.*\bno additional information (?:was|has been) added\b.*$\n?", re.IGNORECASE | re.MULTILINE),
    re.compile(r'^.*\bcorrected\s+".*?"\s+to\s+".*?"\s*\.?\s*$\n?', re.IGNORECASE | re.MULTILINE),
]


def _strip_meta_commentary(text: str) -> str:
    for pattern in _META_COMMENTARY_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def available() -> bool:
    return MODEL_DIR.exists() and any(MODEL_DIR.glob("openvino_model.*"))


def _get_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    try:
        import openvino_genai as ov_genai
        _pipeline = ov_genai.LLMPipeline(str(MODEL_DIR), "NPU")
    except Exception:
        _load_failed = True
        _pipeline = None
    return _pipeline


def format_transcript(class_title: str, session_date: str, transcript: str) -> str | None:
    """Returns clean markdown notes for this transcript, or None if the NPU model isn't
    set up/available/fails for any reason - caller falls back to the heuristic formatter."""
    if not available():
        return None
    pipeline = _get_pipeline()
    if pipeline is None:
        return None

    prompt = f"""Convert this raw speech-to-text lecture transcript into clean, well-organized markdown study notes.

Rules:
- Use bullet points and bold key terms inline - do not add a separate "Key Terms" list
- Silently fix obvious speech-to-text errors - never mention what you corrected or why
- Never add commentary about your own output, your process, or your compliance with these
  rules - your response must contain ONLY the notes themselves, nothing else
- Don't add information that wasn't said

Class: {class_title}

Transcript:
{transcript}

Notes:"""

    try:
        import openvino_genai as ov_genai
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = MAX_NEW_TOKENS
        result = pipeline.generate(prompt, config)
        text = _strip_meta_commentary(str(result))
        if not text:
            return None
        if not text.lstrip().startswith("#"):
            text = f"## {session_date}\n\n{text}"
        return text
    except Exception:
        return None
