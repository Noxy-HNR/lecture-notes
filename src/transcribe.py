"""Local speech-to-text, fully offline, with Cohere Transcribe 03-2026 (2B, Apache 2.0).

In a three-way test on real lectures from this app (PSYC 1300 and BIOL 1440) it disagreed
least with Whisper large-v3 and Qwen3-ASR on both clips, ran several times faster, and got
technical terms right with no vocabulary prompt at all - including "an anion or a cation",
where Whisper wrote "cadmium". It is the only model: the Whisper fallback was removed once
Cohere had run every lecture without failing. Loaded from state/cohere_transcribe/, never
the shared Hugging Face cache (a general cache cleanup has already wiped a model from there
once and stalled a live lecture).

Settings below were tuned against human-verified transcripts (TED-LIUM talks) with
tools/cohere_tuning.py, not against other models' output - see the constants' comments.
Live segments are whole pieces (up to 35s), so windows are handed over back to back.
Optional forced alignment adds word timings to the accepted transcript at session end.
"""
import gc
import os
import re
import sys
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from performance import stage

COHERE_PATH = Path(__file__).resolve().parent.parent / "state" / "cohere_transcribe"

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

_model = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_dependency_lock = threading.Lock()
_dependency_future = None


def _import_dependencies():
    """Import the paired built-in implementations without loading weights or using CUDA."""
    import torch
    if os.environ.get('NOTE_RECORDER_ASR', 'auto') != 'hf':
        return torch, None, None
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    return torch, AutoProcessor, CohereAsrForConditionalGeneration


def start_dependency_preparation() -> Future:
    """Prepare recording dependencies while the action menu waits for an answer.

    The backend joins the same result, including any import error, during preflight.
    A daemon thread lets quitting or choosing another action finish without waiting
    for an unused import. No processor, model weights, or CUDA context is created here.
    Explicit non-recording CLI commands never start this preparation.
    """
    global _dependency_future
    with _dependency_lock:
        if _dependency_future is None:
            future = Future()
            _dependency_future = future

            def prepare():
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    future.set_result(_import_dependencies())
                except BaseException as error:
                    future.set_exception(error)

            threading.Thread(target=prepare, name="transcription-imports", daemon=True).start()
        return _dependency_future

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


def generated_token_count(ids, pad_token_id=None) -> int:
    """Count decoder token IDs, excluding batch padding but retaining real special tokens.

    This measures the model's actual generation work rather than estimating tokens from
    decoded words. Cohere is encoder-decoder, so ``generate()`` returns decoder output
    only; audio/input tokens are not mixed into this count.
    """
    if pad_token_id is None:
        return int(ids.numel() if hasattr(ids, "numel") else np.asarray(ids).size)
    return int((ids != pad_token_id).sum().item())


class _CohereBackend:
    name = "cohere-transcribe"

    def __init__(self):
        torch, AutoProcessor, CohereAsrForConditionalGeneration = start_dependency_preparation().result()
        from health import _read_memory
        if _read_memory()[0] < 2**30:
            raise MemoryError('Less than 1 GiB RAM headroom remains after importing ASR dependencies')
        if AutoProcessor is None:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU (Cohere Transcribe is too slow on CPU)")
        if not (COHERE_PATH / "model.safetensors").exists():
            raise FileNotFoundError(f"model files not found in {COHERE_PATH}")
        free, _ = torch.cuda.mem_get_info()
        if free < (COHERE_PATH / 'model.safetensors').stat().st_size + 2**30:
            raise MemoryError('Insufficient free VRAM for Cohere plus 1 GiB headroom')
        # Processor and model must come from the same implementation. The repo also ships
        # remote-code versions, and pairing the repo's processor with transformers' built-in
        # model fails inside generate() on an unexpected `length` input. Built-in for both.
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(str(COHERE_PATH))
        self.processor.feature_extractor.max_audio_clip_s = COHERE_CLIP_SECONDS
        self.processor.feature_extractor.dither = COHERE_DITHER
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            str(COHERE_PATH), device_map="cuda", dtype=torch.bfloat16)
        self.device_desc = f"cuda/{str(self.model.dtype).removeprefix('torch.')}"
        self.generated_tokens = 0

    def _piece_lengths(self, audio) -> list[int]:
        """The exact split the processor is about to make, so each decoded piece can get its
        own start/end. Uses the feature extractor's own splitter - if a future transformers
        release changes it and the counts stop matching, transcribe() falls back to a single
        segment for the window rather than mislabelling times."""
        fe = self.processor.feature_extractor
        pieces = fe._split_audio_chunks_energy(self._torch.as_tensor(np.asarray(audio, dtype=np.float32)))
        return [int(p.shape[0]) for p in pieces]

    repaired_pieces = 0  # pieces re-decoded by the loop guard, for tools and diagnostics
    silent_pieces = 0    # silent pieces whose invented text the silence guard discarded
    skipped_silent_pieces = 0  # quiet pieces excluded before generation

    def transcribe(self, audio, sample_rate):
        try:
            piece_samples = self._piece_lengths(audio)
        except Exception:
            piece_samples = [len(audio)]
        texts = self._decode(audio, sample_rate, piece_samples=piece_samples)
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
                if text.strip():
                    self.silent_pieces += 1
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

    def _decode(self, audio, sample_rate, *, piece_samples=None) -> list[str]:
        """One text per piece, in order: the processor splits audio longer than a clip at its
        quietest points, and the pieces are decoded COHERE_MAX_BATCH at a time."""
        with stage("cohere_preprocessing"):
            inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", language="en")
        inputs.pop("audio_chunk_index", None)  # one window = one sample; pieces come back in order
        total = inputs["input_features"].shape[0]
        rows = list(range(total))
        # Keep the original feature tensors/padding. Filter only when the exact split
        # matches every feature row; processor changes must never shift timestamps.
        if (piece_samples is not None and len(piece_samples) == total
                and sum(piece_samples) == len(audio) and all(n > 0 for n in piece_samples)):
            rows, cursor = [], 0
            for i, n in enumerate(piece_samples):
                if not is_silent_audio(audio[cursor:cursor + n], sample_rate):
                    rows.append(i)
                cursor += n
        texts = self._decode_rows(inputs, rows)
        if len(rows) == total:
            return texts
        if len(texts) != len(rows):
            # Fail open if a future decoder stops returning one text per row.
            return self._decode_rows(inputs, list(range(total)))
        restored = [""] * total
        for row, text in zip(rows, texts):
            restored[row] = text
        self.skipped_silent_pieces += total - len(rows)
        return restored

    def _decode_rows(self, inputs, rows) -> list[str]:
        torch = self._torch
        texts = []
        for b in range(0, len(rows), COHERE_MAX_BATCH):
            selected = rows[b:b + COHERE_MAX_BATCH]
            batch = {}
            for key, value in inputs.items():
                if not hasattr(value, "shape"):
                    continue
                value = value[selected]
                batch[key] = (value.to(self.model.device, dtype=self.model.dtype)
                              if value.is_floating_point() else value.to(self.model.device))
            with stage("cohere_inference"), torch.inference_mode():
                ids = self.model.generate(**batch, num_beams=COHERE_NUM_BEAMS,
                                          max_new_tokens=int(COHERE_CLIP_SECONDS * 22))
            pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
            self.generated_tokens += generated_token_count(ids, pad_token_id)
            out = self.processor.decode(ids, skip_special_tokens=True)
            texts.extend(out if isinstance(out, list) else [out])
        return texts


class _NanoBackend(_CohereBackend):
    """Same checkpoint and guards, with bounded loading and a pinned lean runtime."""
    def __init__(self):
        torch = start_dependency_preparation().result()[0]
        from health import _read_memory
        if _read_memory()[0] < 2**30 + 128 * 2**20:
            raise MemoryError('Insufficient RAM headroom after importing ASR dependencies')
        if not torch.cuda.is_available():
            raise RuntimeError('no CUDA GPU (Cohere Transcribe is too slow on CPU)')
        for name in ('model.safetensors', 'config.json', 'tokenizer.model'):
            if not (COHERE_PATH / name).is_file():
                raise FileNotFoundError(f'model file not found: {COHERE_PATH / name}')
        from cohere_runtime import WeightLoader, install_graphs
        from vendor.nano_cohere_transcribe._loader import load_model_from_snapshot
        self._torch = torch
        self.loader = WeightLoader()
        self.model, tokenizer = load_model_from_snapshot(
            COHERE_PATH, device='cuda', dtype=torch.bfloat16, tensor_loader=self.loader)
        self.model.tokenizer = tokenizer
        self.model.config.max_audio_clip_s = COHERE_CLIP_SECONDS
        original_prompt = tokenizer.build_prompt
        prefix = tokenizer.sp.piece_to_id('\u2581')
        if prefix == tokenizer.unk_id:
            raise RuntimeError('Cannot reproduce the reference decoder prompt')
        tokenizer.build_prompt = lambda *a, **kw: [prefix] + original_prompt(*a, **kw)
        self.graph_cache = install_graphs(self.model)
        import encoder_runtime
        self.encoder_stats = encoder_runtime.configure(self.model)
        self.generated_tokens = 0
        self.device_desc = f'cuda/bfloat16/nano/{self.loader.stats["mode"]}'
        if self.encoder_stats['mode'] != 'eager':
            self.device_desc += '/' + self.encoder_stats['mode']

    def _pieces(self, audio):
        from vendor.nano_cohere_transcribe.chunk import split_audio_chunks_energy
        return split_audio_chunks_energy(np.asarray(audio, dtype=np.float32), 16000,
                                         COHERE_CLIP_SECONDS, 5.0, 1600)

    def _piece_lengths(self, audio):
        return [len(piece) for piece in self._pieces(audio)]

    def _decode(self, audio, sample_rate, *, piece_samples=None):
        if sample_rate != 16000:
            raise ValueError('Cohere requires 16 kHz audio')
        pieces = self._pieces(audio)
        rows = list(range(len(pieces)))
        if piece_samples == [len(piece) for piece in pieces]:
            rows = [i for i, piece in enumerate(pieces) if not is_silent_audio(piece, sample_rate)]
        before = getattr(self.model, 'generation_tokens', 0)
        try:
            texts = self._generate_pieces(pieces, rows)
        except RuntimeError:
            if not getattr(self.model, 'use_decoder_graphs', True):
                raise
            self.graph_cache.clear()
            self.model.use_decoder_graphs = False
            self.device_desc += '/eager-recovery'
            texts = self._generate_pieces(pieces, rows)
        self.generated_tokens += getattr(self.model, 'generation_tokens', 0) - before
        if len(texts) != len(rows):
            raise RuntimeError('Cohere returned an unexpected number of pieces')
        restored = [''] * len(pieces)
        for row, text in zip(rows, texts):
            restored[row] = text
        self.skipped_silent_pieces += len(pieces) - len(rows)
        return restored

    def _generate_pieces(self, pieces, rows):
        with stage('cohere_inference'), self._torch.inference_mode():
            return self.model._transcribe_chunks_batched(
                [self._torch.from_numpy(pieces[i]) for i in rows], language='en', punctuation=True,
                max_new_tokens=int(COHERE_CLIP_SECONDS * 22), batch_size=COHERE_MAX_BATCH)


def _load_backend():
    mode = os.environ.get('NOTE_RECORDER_ASR', 'auto')
    if mode not in ('auto', 'nano', 'hf'):
        raise ValueError(f'Unknown ASR runtime: {mode}')
    if mode == 'hf':
        return _CohereBackend()
    reason = None
    try:
        return _NanoBackend()
    except Exception as error:
        if mode == 'nano':
            raise
        reason = f'{type(error).__name__}: {error}'
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.cuda.empty_cache()
    backend = _CohereBackend()
    backend.device_desc += f'/reference-fallback ({reason})'
    return backend


def get_model():
    """Loads the model once. A failure raises with the reason, so the startup check can say
    exactly why instead of recording nothing."""
    global _model
    with _model_lock:
        if _model is None:
            try:
                _model = _load_backend()
            except Exception as e:
                raise RuntimeError(f"the transcription model could not load - {type(e).__name__}: {e}") from e
        return _model


def unload_model() -> None:
    """Frees the loaded model (and its VRAM) so the next call loads it fresh."""
    global _model
    with _model_lock:
        _model = None
    gc.collect()
    try:
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.cuda.empty_cache()
    except Exception:
        pass


def backend_name() -> str:
    return get_model().name


def device_info() -> str:
    """Best-effort description of what device/precision the model ended up running on."""
    try:
        backend = get_model()
        if getattr(backend, 'encoder_stats', {}).get('fallbacks'):
            return backend.device_desc + '/encoder-eager-recovery'
        return backend.device_desc
    except Exception:
        return "unknown"


def transcribe_chunk(audio, sample_rate=16000) -> str:
    """audio: 1-D float32 numpy array. Returns transcribed text (may be empty)."""
    segments = transcribe_chunk_segments(audio, sample_rate)
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


def transcribe_chunk_segments(audio, sample_rate=16000) -> list[dict]:
    """Returns a list of {"text", "start", "end"} dicts, start/end in seconds relative to the
    start of `audio` (not the session) - the caller offsets them if needed."""
    model = get_model()
    with _inference_lock:  # one caller on the model at a time (recorder, resume and tools share it)
        segments = model.transcribe(audio, sample_rate)
    return _collapse_repeated_segments(segments)
