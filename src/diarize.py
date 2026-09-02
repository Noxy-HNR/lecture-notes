"""Optional real speaker diarization via pyannote.audio, for labeling who's talking
during Q&A. Fully optional: needs `pip install torch pyannote.audio` (heavy, ~2GB)
plus a free Hugging Face account that has accepted the model's terms, with a token
set as HUGGINGFACE_TOKEN. If any of that isn't set up, diarization is silently
skipped and notes are generated without speaker labels - nothing else breaks.

Setup (one-time):
  1. pip install torch pyannote.audio
  2. Create a free account at https://huggingface.co
  3. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
     and https://huggingface.co/pyannote/segmentation-3.0
  4. Create an access token at https://huggingface.co/settings/tokens
  5. setx HUGGINGFACE_TOKEN "hf_..."
"""
import logging
import os

# Harmless: torch's FLOP counter warns that it can't profile Triton kernels since
# Triton isn't installed (a Linux-focused GPU-kernel compiler, not something this
# app uses). Silence it so it doesn't clutter the console every time torch loads.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

_pipeline = None
_load_failed = False


def available() -> bool:
    return bool(os.environ.get("HUGGINGFACE_TOKEN"))


def _get_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    try:
        from pyannote.audio import Pipeline
        token = os.environ["HUGGINGFACE_TOKEN"]
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token
        )
        try:
            import torch
            if torch.cuda.is_available():
                _pipeline.to(torch.device("cuda"))
        except Exception:
            pass  # CUDA not available/working -> stays on CPU, still functional
    except Exception as e:
        # This used to fail silently every single time (a pyannote.audio version bump
        # renamed from_pretrained's use_auth_token -> token, and the bare except here
        # swallowed the resulting TypeError) - print it so a real setup problem is
        # visible instead of diarization just quietly never doing anything.
        print(f"[diarize] failed to load speaker diarization model: {type(e).__name__}: {e}")
        _load_failed = True
        _pipeline = None
    return _pipeline


def diarize(wav_path) -> list[dict] | None:
    """Returns [{"start": s, "end": s, "speaker": "SPEAKER_00"}, ...] or None if
    diarization isn't set up / available / it failed for this recording."""
    if not available():
        return None
    pipeline = _get_pipeline()
    if pipeline is None:
        return None
    try:
        result = pipeline(str(wav_path))
        # pyannote.audio 4.x wraps the classic Annotation (with .itertracks()) in a
        # DiarizeOutput object under .speaker_diarization; older versions returned the
        # Annotation directly - support both rather than hard-depend on one API shape.
        annotation = getattr(result, "speaker_diarization", result)
        return [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    except Exception as e:
        print(f"[diarize] diarization failed for this clip: {type(e).__name__}: {e}")
        return None


def assign_speakers(transcript_segments: list[dict], diarization_segments: list[dict]) -> list[dict]:
    """transcript_segments: [{"start","end","text"}, ...] with absolute session timestamps.
    Returns the same list with a "speaker" key added (best-overlap match, or None)."""
    labeled = []
    for seg in transcript_segments:
        best_speaker, best_overlap = None, 0.0
        for d in diarization_segments:
            overlap = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, d["speaker"]
        labeled.append({**seg, "speaker": best_speaker})
    return labeled


def to_labeled_transcript(labeled_segments: list[dict]) -> str:
    """Collapses consecutive same-speaker segments into readable '[Speaker N] text' lines.
    Falls back to plain text (no tags) for any segment with no speaker match."""
    lines = []
    current_speaker, current_text = None, []

    def flush():
        if not current_text:
            return
        text = " ".join(current_text)
        if current_speaker:
            n = current_speaker.replace("SPEAKER_", "")
            try:
                label = f"Speaker {int(n) + 1}"
            except ValueError:
                label = current_speaker
            lines.append(f"[{label}] {text}")
        else:
            lines.append(text)

    for seg in labeled_segments:
        if seg["speaker"] != current_speaker:
            flush()
            current_speaker, current_text = seg["speaker"], []
        current_text.append(seg["text"])
    flush()
    return "\n".join(lines)
