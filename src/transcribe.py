"""Local speech-to-text using faster-whisper (fully offline)."""
from faster_whisper import WhisperModel

_model = None

MODEL_SIZE = "large-v3"  # best available accuracy - GPU (RTX 5070 Ti) makes this practical.
                          # Drop to "medium.en" or "small.en" if you ever run this CPU-only.
BEAM_SIZE = 8             # wider search than the default (5) for slightly better decoding
                          # accuracy - cheap given the GPU headroom on this machine.


def get_model():
    global _model
    if _model is None:
        # device="auto" picks the GPU automatically when CUDA + cuDNN are available
        # (much faster), and falls back to CPU otherwise. compute_type="default"
        # picks an appropriate precision per device (float16 on GPU, int8 on CPU).
        try:
            _model = WhisperModel(MODEL_SIZE, device="auto", compute_type="default")
        except Exception:
            # CUDA libs missing/broken -> force CPU rather than crashing.
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def device_info() -> str:
    """Best-effort description of what device/precision Whisper ended up running on."""
    model = get_model()
    try:
        return f"{model.model.device}/{model.model.compute_type}"
    except Exception:
        return "unknown"


def transcribe_chunk(audio, sample_rate=16000, initial_prompt: str | None = None) -> str:
    """audio: 1-D float32 numpy array. Returns transcribed text (may be empty)."""
    segments = transcribe_chunk_segments(audio, sample_rate, initial_prompt)
    return " ".join(seg["text"] for seg in segments).strip()


MAX_CONSECUTIVE_REPEATS = 3  # collapse runs of identical segments longer than this


def _collapse_repeated_segments(segments: list[dict]) -> list[dict]:
    """On ambiguous/overlapping audio (e.g. several people answering quietly at once),
    Whisper can get stuck emitting the same short segment over and over as separate
    consecutive segments (seen live: fifteen consecutive "1." segments, one per VAD
    sub-segment). Whisper's own anti-hallucination heuristics (compression_ratio_threshold,
    log_prob_threshold) don't catch this because each individual short segment looks
    perfectly valid on its own - the repetition is only visible across segments. Cuts
    off a run after MAX_CONSECUTIVE_REPEATS, which still allows a few people genuinely
    giving the same short answer in a row without losing that."""
    if not segments:
        return segments
    result = []
    run_text, run_count = None, 0
    for seg in segments:
        normalized = seg["text"].strip().lower()
        if normalized == run_text:
            run_count += 1
        else:
            run_text, run_count = normalized, 1
        if run_count <= MAX_CONSECUTIVE_REPEATS:
            result.append(seg)
    return result


def transcribe_chunk_segments(audio, sample_rate=16000, initial_prompt: str | None = None) -> list[dict]:
    """Returns a list of {"text", "start", "end"} dicts, start/end in seconds relative
    to the start of `audio` (not the session) - the caller offsets them if needed."""
    model = get_model()
    segments, _info = model.transcribe(
        audio,
        language="en",
        vad_filter=True,          # skip silence instead of hallucinating text
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        beam_size=BEAM_SIZE,
        repetition_penalty=1.1,   # mild nudge against the decoder looping on short phrases
    )
    cleaned = [
        {"text": seg.text.strip(), "start": seg.start, "end": seg.end}
        for seg in segments
        if seg.text.strip()
    ]
    return _collapse_repeated_segments(cleaned)
