"""Local speech-to-text, fully offline.

Two backends, tried in order:

  1. Cohere Transcribe 03-2026 (2B, Apache 2.0) - the default. In a three-way test on real
     lectures from this app (tools/model_ab_test.py, PSYC 1300 and BIOL 1440) it disagreed
     least with the other two models on both clips, ran several times faster, and got
     technical terms right with no vocabulary prompt at all - including "an anion or a
     cation", where Whisper wrote "cadmium". Loaded from state/cohere_transcribe/, never the
     shared Hugging Face cache (a general cache cleanup has already wiped a model from
     there once and stalled a live lecture).
  2. Whisper large-v3 via faster-whisper - the fallback. Used automatically if Cohere can't
     load (model folder missing, no CUDA GPU, transformers not installed), so a missing or
     broken model can never stop a recording from starting.

Cohere settings below were tuned against human-verified transcripts (TED-LIUM talks) with
tools/cohere_tuning.py, not against other models' output - see the constants' comments.

Backend differences callers need to know about:
  - preferred_chunk_seconds(): how much audio to hand over per call. Cohere does best with
    long windows it splits itself at quiet points; Whisper with 20s back-to-back chunks.
  - supports_timestamps(): whether segment timings are fine enough to de-duplicate an audio
    overlap between chunks. Whisper's are; Cohere's segments are whole pieces (up to 35s),
    which is fine for search but not for overlap de-duplication.
  - initial_prompt (the class vocabulary) is used by Whisper and ignored by Cohere, which
    has no prompt or hotword input.
"""
import gc
import re
import threading
from pathlib import Path

import numpy as np
from performance import stage

COHERE_PATH = Path(__file__).resolve().parent.parent / "state" / "cohere_transcribe"

DEFAULT_PREFERENCE = ("cohere", "whisper")
BACKEND_PREFERENCE = DEFAULT_PREFERENCE

# --- Cohere Transcribe, tuned with tools/cohere_tuning.py on TED-LIUM ground truth -------
# Audio handed over per call. Cohere's own splitter then cuts it at the quietest point near
# each COHERE_CLIP_SECONDS boundary, instead of wherever a fixed timer lands mid-word.
# Measured on tuning talks (72 min): fixed 20s cuts 2.88% WER -> 120s windows 2.44%; confirmed
# on held-out talks (76 min): 4.33% -> 3.73%. 300s windows were a tie both times (2.40%,
# 3.67%) but hold live text back 2.5x longer.
COHERE_WINDOW_SECONDS = 120.0
COHERE_CLIP_SECONDS = 35.0      # longest single piece - the model's training clip length
COHERE_NUM_BEAMS = 1            # greedy; beams 2/4/8 no better on tuning talks, 4 worse held-out (3.79%)
COHERE_DITHER = 1e-5            # feature-extractor dither (deterministic, seeded by length)
COHERE_MAX_BATCH = 4            # pieces decoded together - keeps VRAM bounded on long windows

# --- Whisper large-v3 fallback --------------------------------------------------------
MODEL_SIZE = "large-v3"
BEAM_SIZE = 8
WHISPER_CHUNK_SECONDS = 20.0    # measured best of 15/20/25/30s for Whisper (chunk_size_sweep)
# Whether Whisper carries its own previous output forward as context within a chunk. Off:
# prevents one mis-transcription from propagating. Module-level so tools/ can A/B it.
CONDITION_ON_PREVIOUS_TEXT = False

_model = None
_backend = None
_fallback_reason = ""
_inference_lock = threading.Lock()

# --- Cohere failure guards -------------------------------------------------------------
# Two failures seen live in CHEM 1450 on 2026-09-14, then reproduced deterministically in a
# quiet-window benchmark. Neither changes the tuned decoding above: a piece that passes both
# checks keeps exactly the text the normal decode produced.
#
#  1. Digital silence (a muted mic or dropout) comes back as invented sentences: "The world
#     is a very important part of the world." six times over, for pieces that were 100%
#     zeros. Pieces where every SILENCE_SLICE_SECONDS slice is below the app's silence gate
#     get no text.
#  2. Very quiet real room audio (~10% of voice level, class working) can make greedy
#     decoding loop until the token cap: "the other one is the other one ..." for 770 words
#     from 33s of audio. A piece that repeats a phrase REPEAT_LIMIT times in a row, or has
#     more words than anyone can say in its length, is re-decoded in REPAIR_SUB_PIECE_SECONDS
#     sub-pieces, and sub-pieces that still fail are dropped. Global settings were tried
#     first and rejected: a repetition penalty stopped the loop but also changed correct
#     words in normal pieces ("organic classes" -> "organic class with"), and blocking
#     repeated 5-grams left 40 words of garbage in the looping piece.
#
# Thresholds: human-checked TED output never repeated a phrase more than 3 times ("thank
# you" x3), while all six failures in today's session repeated one 4-170 times.
SILENT_PIECE_RMS = 0.002          # same gate as audio.SILENCE_RMS_THRESHOLD
SILENCE_SLICE_SECONDS = 5.0
REPEAT_LIMIT = 4                  # a 2+ word phrase repeated this many times in a row
SINGLE_WORD_REPEAT_LIMIT = 10     # "no, no, no" is speech; ten in a row isn't
MAX_WORDS_PER_SECOND = 8.0        # fast lecturers speak ~3-4; the live loop produced 23
REPAIR_SUB_PIECE_SECONDS = 10.0


def is_silent_audio(samples, sample_rate: int) -> bool:
    """True if every SILENCE_SLICE_SECONDS slice is below SILENT_PIECE_RMS (or it's empty)."""
    step = max(1, int(SILENCE_SLICE_SECONDS * sample_rate))
    return all(float(np.sqrt(np.mean(np.square(samples[i:i + step], dtype=np.float64)))) < SILENT_PIECE_RMS
               for i in range(0, len(samples), step))


def _longest_repeats_by_phrase_length(text: str, max_words: int = 20) -> dict[int, int]:
    """{phrase length in words: most times a phrase of that length repeats back to back}.
    Digits count as words, so "step 1 is, step 2 is, step 3 is" is not "step is" three times."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    result = {}
    for n in range(1, max_words + 1):
        best, i = 1, 0
        while i + 2 * n <= len(words):
            reps = 1
            while words[i + reps * n:i + (reps + 1) * n] == words[i:i + n]:
                reps += 1
            best = max(best, reps)
            i += max(1, (reps - 1) * n)
        result[n] = best
    return result


def looks_degenerate(text: str, seconds: float) -> bool:
    """A decoder loop or runaway output rather than speech - see the guard notes above."""
    words = len(text.split())
    if words > 20 and seconds > 0 and words / seconds > MAX_WORDS_PER_SECOND:
        return True
    for n, reps in _longest_repeats_by_phrase_length(text).items():
        if reps >= (SINGLE_WORD_REPEAT_LIMIT if n == 1 else REPEAT_LIMIT):
            return True
    return False


def segments_from_pieces(texts: list[str], piece_samples: list[int], sample_rate: int) -> list[dict]:
    """Pairs each decoded piece with its time span inside the window, in order. Pieces with
    no text are dropped but still advance the clock, so later timings stay correct."""
    segments, cursor = [], 0
    for text, n in zip(texts, piece_samples):
        start, cursor = cursor, cursor + n
        text = (text or "").strip()
        if text:
            segments.append({"text": text, "start": start / sample_rate, "end": cursor / sample_rate})
    return segments


class _CohereBackend:
    name = "cohere-transcribe"
    timestamps = False
    preferred_chunk_seconds = COHERE_WINDOW_SECONDS

    def __init__(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU (Cohere Transcribe is too slow on CPU)")
        if not (COHERE_PATH / "model.safetensors").exists():
            raise FileNotFoundError(f"model files not found in {COHERE_PATH}")
        # Processor and model must come from the same implementation. The repo also ships
        # remote-code versions, and pairing the repo's processor with transformers' built-in
        # model fails inside generate() on an unexpected `length` input. Built-in for both.
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(str(COHERE_PATH))
        self.processor.feature_extractor.max_audio_clip_s = COHERE_CLIP_SECONDS
        self.processor.feature_extractor.dither = COHERE_DITHER
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            str(COHERE_PATH), device_map="cuda", dtype=torch.bfloat16)
        self.device_desc = f"cuda/{str(self.model.dtype).removeprefix('torch.')}"

    def _piece_lengths(self, audio) -> list[int]:
        """The exact split the processor is about to make, so each decoded piece can get its
        own start/end. Uses the feature extractor's own splitter - if a future transformers
        release changes it and the counts stop matching, transcribe() falls back to a single
        segment for the window rather than mislabelling times."""
        fe = self.processor.feature_extractor
        pieces = fe._split_audio_chunks_energy(self._torch.as_tensor(np.asarray(audio, dtype=np.float32)))
        return [int(p.shape[0]) for p in pieces]

    repaired_pieces = 0  # pieces re-decoded by the loop guard, for tools and diagnostics

    def transcribe(self, audio, sample_rate, initial_prompt=None):
        try:
            piece_samples = self._piece_lengths(audio)
        except Exception:
            piece_samples = [len(audio)]
        texts = self._decode(audio, sample_rate)
        if len(texts) != len(piece_samples):
            text = " ".join(t.strip() for t in texts if t and t.strip())
            return [{"text": text, "start": 0.0, "end": len(audio) / sample_rate}] if text else []
        return segments_from_pieces(self._guard_pieces(audio, sample_rate, texts, piece_samples),
                                    piece_samples, sample_rate)

    def _guard_pieces(self, audio, sample_rate, texts, piece_samples):
        """Applies the two failure guards (see SILENT_PIECE_RMS and REPEAT_LIMIT) piece by
        piece. A piece that passes keeps exactly the text the normal decode produced."""
        guarded, cursor = [], 0
        for text, n in zip(texts, piece_samples):
            piece, cursor = audio[cursor:cursor + n], cursor + n
            if is_silent_audio(piece, sample_rate):
                text = ""
            elif looks_degenerate(text, n / sample_rate):
                self.repaired_pieces += 1
                step = int(REPAIR_SUB_PIECE_SECONDS * sample_rate)
                parts = []
                for j in range(0, n, step):
                    sub = piece[j:j + step]
                    if len(sub) < sample_rate // 2 or is_silent_audio(sub, sample_rate):
                        continue
                    sub_text = " ".join(t.strip() for t in self._decode(sub, sample_rate) if t and t.strip())
                    if sub_text and not looks_degenerate(sub_text, len(sub) / sample_rate):
                        parts.append(sub_text)
                text = " ".join(parts)
            guarded.append(text)
        return guarded

    def _decode(self, audio, sample_rate) -> list[str]:
        """One text per piece, in order: the processor splits audio longer than a clip at its
        quietest points, and the pieces are decoded COHERE_MAX_BATCH at a time."""
        torch = self._torch
        with stage("cohere_preprocessing"):
            inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", language="en")
        inputs.pop("audio_chunk_index", None)  # one window = one sample; pieces come back in order
        texts = []
        total = inputs["input_features"].shape[0]
        for b in range(0, total, COHERE_MAX_BATCH):
            batch = {}
            for key, value in inputs.items():
                if not hasattr(value, "shape"):
                    continue
                value = value[b:b + COHERE_MAX_BATCH]
                batch[key] = (value.to(self.model.device, dtype=self.model.dtype)
                              if value.is_floating_point() else value.to(self.model.device))
            with stage("cohere_inference"), torch.inference_mode():
                ids = self.model.generate(**batch, num_beams=COHERE_NUM_BEAMS,
                                          max_new_tokens=int(COHERE_CLIP_SECONDS * 22))
            out = self.processor.decode(ids, skip_special_tokens=True)
            texts.extend(out if isinstance(out, list) else [out])
        return texts


class _WhisperBackend:
    name = "whisper-large-v3"
    timestamps = True
    preferred_chunk_seconds = WHISPER_CHUNK_SECONDS

    def __init__(self):
        from faster_whisper import WhisperModel
        try:
            # device="auto" picks the GPU when CUDA + cuDNN are available; compute_type
            # "default" picks float16 on GPU, int8 on CPU.
            self.model = WhisperModel(MODEL_SIZE, device="auto", compute_type="default")
        except Exception:
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        try:
            self.device_desc = f"{self.model.model.device}/{self.model.model.compute_type}"
        except Exception:
            self.device_desc = "unknown"

    def transcribe(self, audio, sample_rate, initial_prompt=None):
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,          # skip silence instead of hallucinating text
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            initial_prompt=initial_prompt,
            beam_size=BEAM_SIZE,
            repetition_penalty=1.1,   # mild nudge against the decoder looping on short phrases
        )
        return [{"text": seg.text.strip(), "start": seg.start, "end": seg.end}
                for seg in segments if seg.text.strip()]


_LOADERS = {"cohere": _CohereBackend, "whisper": _WhisperBackend}


def get_model():
    """Loads the first backend in BACKEND_PREFERENCE that works, once. If a preferred one
    fails, the reason is kept for fallback_reason() so the app can say so rather than
    silently running on a different model."""
    global _model, _backend, _fallback_reason
    if _model is not None:
        return _model
    failures = []
    for name in BACKEND_PREFERENCE:
        try:
            _model = _LOADERS[name]()
            _backend = name
            break
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
    if _model is None:
        raise RuntimeError("no transcription model could load - " + "; ".join(failures))
    _fallback_reason = "; ".join(failures)
    return _model


def use_backend(name: str | None) -> None:
    """Pins one backend (or restores the default order with None) and unloads whatever is
    loaded. For tools/ that tune one specific model and must not silently test another."""
    global _model, _backend, _fallback_reason, BACKEND_PREFERENCE
    _model, _backend, _fallback_reason = None, None, ""
    BACKEND_PREFERENCE = (name,) if name else DEFAULT_PREFERENCE
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def backend_name() -> str:
    return get_model().name


def fallback_reason() -> str:
    """Non-empty when a preferred backend failed to load and a fallback is in use."""
    get_model()
    return _fallback_reason


def supports_timestamps() -> bool:
    return get_model().timestamps


def preferred_chunk_seconds() -> float:
    """Seconds of audio to hand the loaded backend per call (each was tuned separately)."""
    return float(getattr(get_model(), "preferred_chunk_seconds", WHISPER_CHUNK_SECONDS))


def device_info() -> str:
    """Best-effort description of what device/precision the model ended up running on."""
    try:
        return get_model().device_desc
    except Exception:
        return "unknown"


def transcribe_chunk(audio, sample_rate=16000, initial_prompt: str | None = None) -> str:
    """audio: 1-D float32 numpy array. Returns transcribed text (may be empty)."""
    segments = transcribe_chunk_segments(audio, sample_rate, initial_prompt)
    return " ".join(seg["text"] for seg in segments).strip()


MAX_CONSECUTIVE_REPEATS = 3  # collapse runs of identical segments longer than this


def _collapse_repeated_segments(segments: list[dict]) -> list[dict]:
    """On ambiguous/overlapping audio (e.g. several people answering quietly at once), a
    model can get stuck emitting the same short segment over and over (seen live: fifteen
    consecutive "1." segments). The repetition is only visible across segments, so the
    per-segment anti-hallucination heuristics don't catch it. Cuts off a run after
    MAX_CONSECUTIVE_REPEATS, which still allows a few people genuinely giving the same
    short answer in a row."""
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
    """Returns a list of {"text", "start", "end"} dicts, start/end in seconds relative to the
    start of `audio` (not the session) - the caller offsets them if needed."""
    model = get_model()
    with _inference_lock:  # one caller on the model at a time (recorder, resume and tools share it)
        segments = model.transcribe(audio, sample_rate, initial_prompt)
    return _collapse_repeated_segments(segments)
