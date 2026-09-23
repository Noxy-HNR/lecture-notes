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
(a few seconds - no NPU-style ahead-of-time compilation). An idle-shutdown watchdog
stops it automatically after IDLE_TIMEOUT_SECONDS of disuse, so a session where this
tier is only ever a transient fallback (Claude CLI/API hiccup, then recovers) doesn't
keep it holding ~3.4GB of VRAM for the rest of the session; it starts right back up on
the next call if needed. Call stop_server() directly to shut it down immediately (e.g.
at app exit, via the atexit hook this module registers on startup).

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
import ctypes
import json
import os
import subprocess
import threading
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

# How long the server can sit unused before the idle watchdog shuts it down. This tier
# is usually only a transient fallback (Claude CLI/API hiccups, then recovers) - without
# this, one brief fallback anywhere in a session left it holding ~3.4GB of VRAM and idle
# GPU power draw for the rest of that session even after CLI/API were working fine again
# for everything since. 10 minutes is comfortably longer than a real autosave interval
# (5 min), so a session actually still using this tier regularly never gets interrupted.
IDLE_TIMEOUT_SECONDS = 600
_WATCHDOG_POLL_SECONDS = 60  # module-level so tests can shrink both this and the
                             # timeout above to verify real shutdown timing quickly,
                             # without waiting out the real 10-minute window

_server_process = None
_server_job = None
_last_used = 0.0
_watchdog_stop_event = None
_atexit_registered = False
_server_lock = threading.RLock()


def _attach_windows_kill_job(process):
    """Tie an owned server to this Python process at the Windows kernel level.

    ``atexit`` is not guaranteed to run when somebody closes the recorder's console
    window.  A Job Object with KILL_ON_JOB_CLOSE is: Windows closes our non-inherited
    job handle when this process disappears and then terminates the llama-server child.
    Only the exact ``Popen`` child created below is assigned, so a separately launched
    server (including the user's server on port 10000) can never be affected.

    Returns the job handle, or ``None`` when unavailable.  Explicit shutdown and the
    atexit hook remain in place as the portable/graceful cleanup paths.
    """
    if os.name != "nt":
        return None

    from ctypes import wintypes

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    assigned = configured and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle))
    if not assigned:
        kernel32.CloseHandle(job)
        return None
    return job


def _close_windows_job(job):
    if job is not None and os.name == "nt":
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)


def available() -> bool:
    return LLAMA_SERVER_EXE.exists() and MODEL_PATH.exists()


def warm_up() -> bool:
    """Starts llama-server now instead of waiting for the first real format_transcript()/
    condense_transcript() call to lazily do it. Worth calling during preflight only in
    "local" formatting mode, where this tier is guaranteed to be needed eventually - in
    "auto" mode it's usually never touched (Claude CLI/API handle everything), so
    pre-warming there would just burn ~3.4GB VRAM and ~30s of startup time for nothing in
    the common case. Returns whether the server ended up healthy."""
    return _ensure_server_running()


def _server_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def _touch():
    """Marks the server as just-used, resetting the idle-shutdown clock. Called both
    when the server is confirmed up (warm_up() or the first real call starting it - so
    the idle window starts counting from "just started", not from a stale/zero
    timestamp that would let the watchdog kill it before it's ever actually used) and
    on every real _chat() call (so genuine ongoing use keeps it alive)."""
    global _last_used
    _last_used = time.time()


def _start_idle_watchdog():
    """One watchdog thread per server start, stopped and replaced whenever the server
    is (re)started - not a single long-lived thread, since a plain threading.Event can
    only ever transition low->high once and needs to be fresh for each server
    lifetime."""
    global _watchdog_stop_event
    stop_event = threading.Event()
    _watchdog_stop_event = stop_event

    def _loop():
        # wait() returns True (and exits the loop) as soon as stop_event is set, or
        # False after each poll interval - so this reacts to stop_server() promptly
        # instead of sleeping through it.
        while not stop_event.wait(_WATCHDOG_POLL_SECONDS):
            if time.time() - _last_used > IDLE_TIMEOUT_SECONDS:
                stop_server()
                return

    threading.Thread(target=_loop, daemon=True, name="gpu-formatter-idle-watchdog").start()


def _ensure_server_running() -> bool:
    global _server_process, _server_job, _atexit_registered
    with _server_lock:
        if _server_healthy():
            _touch()
            return True
        if not available():
            return False
        if _server_process is not None and _server_process.poll() is not None:
            _close_windows_job(_server_job)
            _server_job = None
            _server_process = None  # previous process died - allow restarting

        if _server_process is None:
            try:
                _server_process = subprocess.Popen(
                    [str(LLAMA_SERVER_EXE), "-m", str(MODEL_PATH), "--port", str(PORT),
                     "-ngl", "999", "-c", str(CONTEXT_SIZE), "--log-disable"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                _server_job = _attach_windows_kill_job(_server_process)
                if os.name == "nt" and _server_job is None:
                    # Do not run an owned Windows server without the abrupt-close
                    # guarantee. The caller can safely use the heuristic fallback.
                    stop_server()
                    return False
                if not _atexit_registered:
                    atexit.register(stop_server)
                    _atexit_registered = True
            except Exception:
                stop_server()
                return False

        deadline = time.time() + STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if _server_healthy():
                _touch()
                _start_idle_watchdog()
                return True
            time.sleep(0.5)
        stop_server()  # failed startup must not leave a half-loaded model process
        return False


def stop_server():
    """Stop only the server process this module launched; borrowed servers are untouched."""
    global _server_process, _server_job, _watchdog_stop_event
    with _server_lock:
        if _watchdog_stop_event is not None:
            _watchdog_stop_event.set()
            _watchdog_stop_event = None
        process = _server_process
        job = _server_job
        _server_process = None
        _server_job = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        _close_windows_job(job)


def _chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    if not _ensure_server_running():
        return None
    _touch()  # real use - covers the case where the server was already running/warmed
              # up (the _ensure_server_running() early-return path doesn't touch it itself)
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
