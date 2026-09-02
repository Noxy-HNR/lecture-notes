"""Local, offline note formatting using a local LLM (Qwen2.5-3B-Instruct, GGUF)
running on the GPU via llama.cpp's llama-server, reached over HTTP. Sits between the
Claude CLI/API and the pure heuristic formatter in notes.py's fallback chain.

Uses the machine's already-installed llama.cpp binary
(C:/AI/Tools/llama-native/bin/llama-server.exe) read-only, as a completely separate
process on its own port (8090) - does NOT touch, reuse, or reconfigure the user's
personal llama.cpp server/model setup on port 10000. The model file lives in this
project's own state/llama_model/, never in the user's models/ folder.

Setup (one-time):
  Download the model into this project (NOT the user's models/ folder):
    from huggingface_hub import hf_hub_download
    hf_hub_download("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q8_0.gguf",
                     local_dir="state/llama_model")

format_transcript()/condense_transcript() start the server automatically on first use
(a few seconds - no NPU-style ahead-of-time compilation) and reuse it for the rest of
the process's lifetime. Call stop_server() to shut it down (e.g. at app exit) so it
doesn't sit in the background holding ~3.6GB of VRAM after the app closes.

Model notes: this replaced an earlier NPU-based tier (Phi-3.5-mini via OpenVINO) that
was rolled back after repeated reliability failures on the real pipeline - degenerate
repetition with default (greedy) decoding, and hallucinated/incoherent rambling once a
repetition_penalty was added to fix that. Root cause was likely a mix of the small
NPU-quantized model plus OpenVINO GenAI's decoding defaults; llama.cpp's more mature
sampling stack (temperature + repeat_penalty tuned together) on a larger 3B model
running on the actual discrete GPU (OpenVINO's "GPU" device only targets Intel
integrated graphics, not NVIDIA - a separate discovery) has been reliable in testing.
"""
import atexit
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_SERVER_EXE = Path("C:/AI/Tools/llama-native/bin/llama-server.exe")
MODEL_PATH = PROJECT_ROOT / "state" / "llama_model" / "qwen2.5-3b-instruct-q8_0.gguf"
PORT = 8090  # distinct from the user's existing llama-server on port 10000
BASE_URL = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_SECONDS = 30

# Was 4096 - way too small for this tier's real job. It's the fallback for a whole
# SESSION's transcript (via try_full_session_diarization in main.py) when Claude
# CLI/API aren't available, not just a single ~15s chunk. Confirmed live: a real
# 37-minute lecture (~7,350 tokens) silently failed to fit and fell all the way
# through to the much lower-quality heuristic formatter instead of this tier - not
# a crash, just silent degradation, which is worse because nothing looked broken.
# 32768 comfortably covers a 3+ hour lecture (Qwen2.5-3B supports up to 32k context).
CONTEXT_SIZE = 32768
REQUEST_TIMEOUT_SECONDS = 180  # was 60 - too tight once output scales with input (below)

_server_process = None


def available() -> bool:
    return LLAMA_SERVER_EXE.exists() and MODEL_PATH.exists()


def _server_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def _ensure_server_running() -> bool:
    global _server_process
    if _server_healthy():
        return True
    if not available():
        return False
    if _server_process is not None and _server_process.poll() is not None:
        _server_process = None  # previous process died - allow restarting

    if _server_process is None:
        try:
            _server_process = subprocess.Popen(
                [str(LLAMA_SERVER_EXE), "-m", str(MODEL_PATH), "--port", str(PORT),
                 "-ngl", "999", "-c", str(CONTEXT_SIZE), "--log-disable"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            atexit.register(stop_server)
        except Exception:
            return False

    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if _server_healthy():
            return True
        time.sleep(0.5)
    return False


def stop_server():
    global _server_process
    if _server_process is not None and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
    _server_process = None


def _chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    if not _ensure_server_running():
        return None
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "repeat_penalty": 1.1,
    }
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None


_FORMAT_SYSTEM_PROMPT = """Convert raw speech-to-text lecture transcripts into clean, well-organized markdown study notes.

Rules:
- Write full, clear bullet points (not bare keyword fragments) - each bullet should be a complete thought
- Bold the key terms within each bullet using **term** - do not add a separate "Key Terms" list
- Silently fix obvious speech-to-text errors (misspelled technical terms, etc.) - never mention what you corrected
- Never add commentary about your own output or process - respond with ONLY the notes
- Don't add information that wasn't said in the transcript

Example of the style to match:
Transcript: "the mitocondria is the powerhouse of the cell. it produces energy through a process called cellular respiration."
Notes:
- **Mitochondria**: known as the "powerhouse of the cell"
- Produces energy through **cellular respiration**"""


def _scaled_max_tokens(transcript: str, floor: int = 400, ceiling: int = 4000) -> int:
    """Notes output needs to scale with input length - a fixed small cap (this used
    to be a flat 400) truncates output for anything longer than a short chunk, most
    obviously a whole-session transcript. Rough heuristic: notes run shorter than the
    source transcript, so half the estimated input token count is a reasonable upper
    bound, clamped to a sane range either way."""
    estimated_input_tokens = len(transcript.split()) * 1.3
    return int(min(ceiling, max(floor, estimated_input_tokens * 0.5)))


def format_transcript(class_title: str, session_date: str, transcript: str) -> str | None:
    """Returns clean markdown notes for this transcript, or None if the local LLM
    server isn't available/fails - caller falls back to the heuristic formatter."""
    user_prompt = f"Class: {class_title}\n\nTranscript:\n{transcript}"
    text = _chat(_FORMAT_SYSTEM_PROMPT, user_prompt, max_tokens=_scaled_max_tokens(transcript))
    if not text:
        return None
    if not text.lstrip().startswith("#"):
        text = f"## {session_date}\n\n{text}"
    return text


_CONDENSE_SYSTEM_PROMPT = """You are cleaning up notes from ONE lecture recording session that were saved
multiple times (autosaves), which can leave duplicate or overlapping sections.

Rules:
- Merge into exactly ONE "## <date>" section - reuse the existing date, don't invent one
- Merge duplicate/overlapping bullets into a single clean bullet - don't just concatenate them
- CRITICAL: every bullet in the input covers a distinct piece of information UNLESS it is
  clearly restating the same fact as another bullet. Before finishing, check each input
  bullet individually and confirm it is represented in your output somewhere - do not
  drop a bullet just because it's short, or because it appeared in a section by itself
- Fix broken markdown formatting
- If a "### Q&A" section is present in the notes below, keep it separate, cleaned up the same way.
  If there is no Q&A content in the notes below, do not create one.
- Preserve any "[FLAG: ...]" or "verify" markers on the point they're attached to
- Do not invent new content or new sections
- Respond with ONLY the final markdown, no preamble or explanation, no notes about what you did"""


def condense_transcript(session_markdown: str) -> str | None:
    """Re-reviews notes from one lecture session (possibly containing duplicate/
    overlapping sections from multiple autosaves) and merges them into one clean
    section. Returns None if the local LLM server isn't available/fails. Currently
    unused (notes.py's condense_session is CLI/API-only - see its docstring for why),
    kept in sync with the same context/token-scaling fix as format_transcript in case
    that decision is ever revisited."""
    return _chat(_CONDENSE_SYSTEM_PROMPT, session_markdown, max_tokens=_scaled_max_tokens(session_markdown, ceiling=3000))
