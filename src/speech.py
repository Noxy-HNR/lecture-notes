"""Offline narration for lessons, using Kokoro-82M through kokoro-onnx.

Kokoro was picked over Windows' built-in SAPI voices (robotic) and Piper (smaller but
noticeably flatter) because lessons are listened to for minutes at a time. It runs on
the CPU through onnxruntime, so it never competes with transcription for the GPU.

Model files live in state/kokoro/ (not the shared Hugging Face cache, which a cleanup
has wiped before). One-time download, about 340 MB:

    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
"""
import os
import re
import threading
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "state" / "kokoro"
MODEL_PATH = MODEL_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = MODEL_DIR / "voices-v1.0.bin"

VOICES = {
    "af_heart": "Heart - US, warm",
    "af_bella": "Bella - US, bright",
    "am_michael": "Michael - US",
    "bf_emma": "Emma - UK",
    "bm_george": "George - UK",
}
DEFAULT_VOICE = "af_heart"

# Narration is spoken in sentence-sized pieces and joined with short pauses. Kokoro's model
# takes at most ~510 phoneme tokens per call, and pieces keep every call well inside that
# for a 60-150 word slide narration.
MAX_CHUNK_CHARS = 250
CHUNK_PAUSE_SECONDS = 0.18
# libsndfile's Vorbis encoder allocates per-write buffers on the stack: handing it ~25s or
# more of audio in one write crashed the whole process with a native stack overflow
# (0xC00000FD, no Python error). Audio is written in small blocks instead.
WRITE_BLOCK_FRAMES = 8192
# onnxruntime's default spreads work over every logical core. On this laptop's hybrid CPU
# (8 performance + 12 efficiency cores) that ran at 0.96s of work per second of speech;
# 8 threads took 0.50s and 16 took 0.61s. The int8 model was slower still (1.75-2.91s).
SYNTHESIS_THREADS = min(8, os.cpu_count() or 8)

_engine = None
_engine_lock = threading.Lock()  # one synthesis at a time; the ONNX session isn't shared safely


def available() -> tuple[bool, str]:
    """(usable, reason) - lessons are still built without narration when this is False."""
    if not MODEL_PATH.exists() or not VOICES_PATH.exists():
        return False, f"voice model files not found in {MODEL_DIR}"
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        return False, "kokoro-onnx is not installed"
    return True, ""


def speakable(text: str) -> str:
    """Strips what a voice would read literally: markdown emphasis, bullets, arrows."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = text.replace("→", " to ").replace("←", " from ").replace("⇌", " is in equilibrium with ")
    text = text.replace("&", " and ").replace("%", " percent")
    return re.sub(r"\s+", " ", text).strip()


def _load_engine():
    from kokoro_onnx import Kokoro
    try:
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = SYNTHESIS_THREADS
        session = onnxruntime.InferenceSession(str(MODEL_PATH), sess_options=options,
                                               providers=["CPUExecutionProvider"])
        return Kokoro.from_session(session, str(VOICES_PATH))
    except (ImportError, AttributeError):
        return Kokoro(str(MODEL_PATH), str(VOICES_PATH))


def chunks(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Splits text into pieces of at most `limit` characters, preferring sentence breaks,
    then commas, then spaces, and packing short neighbouring sentences together."""
    pieces = []
    for sentence in re.split(r"(?<=[.!?;:])\s+", text.strip()):
        while len(sentence) > limit:
            cut = sentence.rfind(", ", 0, limit)
            if cut < limit // 3:
                cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit - 1
            pieces.append(sentence[:cut + 1].strip())
            sentence = sentence[cut + 1:].strip()
        if sentence:
            pieces.append(sentence)
    packed = []
    for piece in pieces:
        if packed and len(packed[-1]) + 1 + len(piece) <= limit:
            packed[-1] += " " + piece
        else:
            packed.append(piece)
    return packed


def synthesize(text: str, out_path: Path, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> float:
    """Writes `text` as spoken audio (OGG Vorbis) and returns its length in seconds."""
    global _engine
    import numpy as np
    import soundfile as sf
    voice = voice if voice in VOICES else DEFAULT_VOICE
    parts, sample_rate = [], 24000
    with _engine_lock:
        if _engine is None:
            _engine = _load_engine()
        for piece in chunks(speakable(text)):
            samples, sample_rate = _engine.create(piece, voice=voice, speed=speed, lang="en-us")
            if parts:
                parts.append(np.zeros(int(CHUNK_PAUSE_SECONDS * sample_rate), dtype=np.float32))
            parts.append(np.asarray(samples, dtype=np.float32))
    if not parts:
        raise ValueError("nothing to say")
    audio = np.concatenate(parts)
    with sf.SoundFile(str(out_path), "w", samplerate=sample_rate, channels=1,
                      format="OGG", subtype="VORBIS") as handle:
        for start in range(0, len(audio), WRITE_BLOCK_FRAMES):
            handle.write(audio[start:start + WRITE_BLOCK_FRAMES])
    return len(audio) / sample_rate
