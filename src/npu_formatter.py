"""Local, offline note formatting using a small LLM (Qwen2.5-1.5B-Instruct) running on
the machine's NPU (Intel AI Boost) via OpenVINO GenAI. Sits between the Claude CLI/API
and the pure heuristic formatter in notes.py's fallback chain: much more capable than
regex-based formatting (actual rewriting/cleanup, not just bullet-splitting) while
staying fully local, fast, and off the GPU that's busy transcribing.

Optional setup (see README "NPU-accelerated local formatting"):
  1. pip install openvino openvino-tokenizers openvino-genai
  2. Download the model once:
       from huggingface_hub import snapshot_download
       snapshot_download("OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", local_dir="state/npu_model")

If the packages aren't installed or the model isn't downloaded, format_transcript()
returns None and notes.py falls back further to the pure heuristic formatter - nothing
else breaks.
"""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "state" / "npu_model"
MAX_NEW_TOKENS = 800

_pipeline = None
_load_failed = False


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
Use bullet points and bold the key terms. Fix obvious speech-to-text errors, but don't add information that wasn't said.
Output ONLY the notes, no preamble or explanation.

Class: {class_title}

Transcript:
{transcript}

Notes:"""

    try:
        import openvino_genai as ov_genai
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = MAX_NEW_TOKENS
        result = pipeline.generate(prompt, config)
        text = str(result).strip()
        if not text:
            return None
        if not text.lstrip().startswith("#"):
            text = f"## {session_date}\n\n{text}"
        return text
    except Exception:
        return None
