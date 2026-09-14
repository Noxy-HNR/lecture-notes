"""Head-to-head ASR comparison on this user's real lectures:
Whisper large-v3 vs Qwen3-ASR 1.7B vs Cohere Transcribe 03-2026.

METHOD NOTE - why this doesn't report WER against a reference:
there is no human ground truth for these lectures. Scoring each model against its own
single-pass output (what the other harnesses here do) only works within one model - across
models every model would grade itself best. So this measures what doesn't need an answer
key:

  1. Peer divergence - each model's average WER against the OTHER two. With three
     independent models, the one that disagrees most with both others is the likeliest to
     be wrong, since two models rarely make the same mistake on the same word. A proxy,
     not a score - but a much better one than two models just disagreeing with each other.
  2. Dropped content - word count and truncated thoughts ("..."). An LLM can clean up
     choppy punctuation downstream; it cannot recover words that were never transcribed.
  3. Degenerate output - longest immediately-repeating phrase run (looping).
  4. Cost - wall time, realtime factor, peak VRAM (torch-allocated only - Whisper runs on
     CTranslate2, which allocates outside torch, so its VRAM reads ~0).

All three models get identical audio chunks. Whisper and Qwen also get the class vocabulary
from vocab.json; Cohere Transcribe has no prompt/hotword support at all, so it runs without
- which is also how it would have to be used for real. Transcripts are written out, because
reading the disagreements side by side is what actually settles quality.

Refuses to run while a live recording is in progress (state/status.json heartbeat): three
~4GB models competing for the GPU mid-lecture could stall or crash the recorder.

    python tools/model_ab_test.py <wav> [<wav> ...] [--seconds 300] [--chunk 20]
                                  [--models whisper qwen cohere] [--force]
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf

import audio as audio_mod
import vocab as vocab_module
from vad_ab_test import word_error_rate

ROOT = Path(__file__).resolve().parent.parent
QWEN_PATH = ROOT / "state" / "qwen3_asr"
COHERE_PATH = ROOT / "state" / "cohere_transcribe"
COHERE_REPO = "CohereLabs/cohere-transcribe-03-2026"
OUT_DIR = ROOT / "state" / "model_ab"
STATUS_PATH = ROOT / "state" / "status.json"


def recording_in_progress() -> bool:
    try:
        s = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return bool(s.get("active")) and (time.time() - s.get("updated_at", 0)) < 90
    except Exception:
        return False


def class_code_from(wav: Path) -> str:
    m = re.match(r"^(.+)_\d{8}_\d{6}$", wav.stem)
    return m.group(1).replace("_", " ") if m else ""


def chunks_of(samples: np.ndarray, seconds: float) -> list[np.ndarray]:
    step = int(seconds * audio_mod.SAMPLE_RATE)
    return [samples[i:i + step] for i in range(0, len(samples), step)]


def _free_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# The three models, each behind the same interface: (chunks, vocab_prompt) ->
# (text, seconds, peak_vram_gb). Models load/unload one at a time - 12GB won't
# comfortably hold more than one plus headroom.
# ---------------------------------------------------------------------------

def run_whisper(chunks, vocab_prompt):
    import torch
    import transcribe
    transcribe.use_backend("whisper")  # the app defaults to Cohere now; this run must be Whisper
    transcribe.get_model()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = []
    for chunk in chunks:
        segs = transcribe.transcribe_chunk_segments(chunk, initial_prompt=vocab_prompt)
        out.extend(s["text"] for s in segs)
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    transcribe.use_backend(None)  # unload Whisper and restore the app's default order
    _free_gpu()
    return " ".join(out).strip(), elapsed, peak


def run_qwen(chunks, vocab_prompt):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    proc = AutoProcessor.from_pretrained(str(QWEN_PATH))
    model = AutoModelForMultimodalLM.from_pretrained(
        str(QWEN_PATH), device_map="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    prompt = f"Vocabulary: {vocab_prompt}" if vocab_prompt else None
    t0 = time.time()
    out = []
    for chunk in chunks:
        inputs = proc.apply_transcription_request(
            audio=chunk, prompt=prompt, language="English").to(model.device, model.dtype)
        ids = model.generate(**inputs, max_new_tokens=440)
        gen = ids[:, inputs["input_ids"].shape[1]:]
        out.append(proc.decode(gen, return_format="transcription_only")[0])
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, proc
    _free_gpu()
    return " ".join(out).strip(), elapsed, peak


def ensure_cohere_downloaded():
    """Gated repo: the HF account behind HUGGINGFACE_TOKEN must have accepted the terms on
    the model page first. Downloads into this project's own state/ - never the shared
    ~/.cache/huggingface/hub, which a general cache cleanup has already wiped once."""
    if (COHERE_PATH / "model.safetensors").exists():
        return
    import os
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    snapshot_download(COHERE_REPO, local_dir=str(COHERE_PATH), token=token)


def run_cohere(chunks, vocab_prompt):
    """vocab_prompt is intentionally unused: Cohere Transcribe has no prompt, context or
    hotword input - language is the only conditioning it accepts."""
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    ensure_cohere_downloaded()
    # Processor and model MUST come from the same implementation. The repo ships its own
    # remote-code processor, which emits `length`; transformers 5.x has a built-in model
    # class that takes `attention_mask` instead and rejects `length` outright. Mixing the
    # two (repo processor + built-in model, as the model card's snippet effectively does
    # on this transformers version) fails inside generate(). Both built-in here, no
    # trust_remote_code at all.
    proc = AutoProcessor.from_pretrained(str(COHERE_PATH))
    model = CohereAsrForConditionalGeneration.from_pretrained(
        str(COHERE_PATH), device_map="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = []
    for chunk in chunks:
        inputs = proc(chunk, sampling_rate=audio_mod.SAMPLE_RATE, return_tensors="pt", language="en")
        # Bookkeeping for reassembling long audio, not a model input - generate() rejects
        # unknown kwargs, so it goes to decode() instead.
        chunk_index = inputs.pop("audio_chunk_index", None)
        inputs = inputs.to(model.device, dtype=model.dtype)
        ids = model.generate(**inputs, max_new_tokens=440)
        decode_kwargs = {"skip_special_tokens": True}
        if chunk_index is not None:
            decode_kwargs.update(audio_chunk_index=chunk_index, language="en")
        text = proc.decode(ids, **decode_kwargs)
        out.append(text[0] if isinstance(text, list) else text)
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, proc
    _free_gpu()
    return " ".join(t.strip() for t in out).strip(), elapsed, peak


RUNNERS = {
    "whisper": ("whisper-large-v3", run_whisper),
    "qwen": ("qwen3-asr-1.7b", run_qwen),
    "cohere": ("cohere-transcribe", run_cohere),
}


# ---------------------------------------------------------------------------
# Ground-truth-free metrics
# ---------------------------------------------------------------------------

def longest_repeat_run(text: str, n: int = 6) -> int:
    words = text.lower().split()
    if len(words) < n * 2:
        return 0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    best = run = 1
    for a, b in zip(grams, grams[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _norm(text: str) -> str:
    """Punctuation and case are formatting choices, not recognition errors - strip them so
    divergence measures which WORDS each model heard."""
    return re.sub(r"[^\w\s']", " ", text.lower())


def symmetric_wer(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    return (word_error_rate(a, b) + word_error_rate(b, a)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wavs", nargs="+")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--chunk", type=float, default=20.0)
    parser.add_argument("--models", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    parser.add_argument("--force", action="store_true", help="run even if a recording looks live")
    args = parser.parse_args()

    if recording_in_progress() and not args.force:
        print("A recording is in progress (state/status.json heartbeat is fresh). Not loading "
              "models onto the GPU mid-lecture. Stop the recorder first, or pass --force.")
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for wav_path in args.wavs:
        wav = Path(wav_path)
        code = class_code_from(wav)
        vocab_prompt = vocab_module.initial_prompt_for_class(code)
        with sf.SoundFile(str(wav)) as f:
            samples = f.read(int(args.seconds * audio_mod.SAMPLE_RATE), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        clip_seconds = len(samples) / audio_mod.SAMPLE_RATE
        chunks = chunks_of(samples, args.chunk)

        print(f"\n{'=' * 78}\n{wav.name} | {code} | {clip_seconds:.0f}s | "
              f"{len(chunks)} x {args.chunk:.0f}s chunks\n{'=' * 78}", flush=True)

        results = {}
        for key in args.models:
            name, runner = RUNNERS[key]
            print(f"  running {name} ...", flush=True)
            try:
                text, elapsed, peak = runner(chunks, vocab_prompt)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            results[name] = {"text": text, "seconds": elapsed, "peak_vram": peak,
                              "words": len(text.split()), "ellipses": text.count("..."),
                              "repeat_run": longest_repeat_run(text), "rtf": elapsed / clip_seconds}
            (OUT_DIR / f"{wav.stem}__{name}.txt").write_text(text, encoding="utf-8")
            r = results[name]
            print(f"    {r['words']:>5} words | {elapsed:>6.1f}s ({r['rtf']:.2f}x realtime) | "
                  f"{peak:.2f}GB | truncations {r['ellipses']} | repeat run {r['repeat_run']}", flush=True)

        names = list(results)
        pair = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair[(a, b)] = pair[(b, a)] = symmetric_wer(results[a]["text"], results[b]["text"])
        if len(names) >= 2:
            print("\n  pairwise disagreement (words only, punctuation/case ignored):")
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    print(f"    {a:<20} vs {b:<20} {pair[(a, b)]:.1%}")
        if len(names) >= 3:
            print("\n  peer divergence (avg disagreement with the OTHER models - lower = likelier right):")
            for a in sorted(names, key=lambda n: np.mean([pair[(n, o)] for o in names if o != n])):
                results[a]["peer_divergence"] = float(np.mean([pair[(a, o)] for o in names if o != a]))
                print(f"    {a:<20} {results[a]['peer_divergence']:.1%}")
        summary[wav.name] = results

    print(f"\n\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'clip':<32}{'model':<20}{'peer div':>9}{'words':>7}{'trunc':>7}{'rtf':>7}{'rep':>5}")
    for clip, results in summary.items():
        for name, r in results.items():
            pdv = f"{r['peer_divergence']:.1%}" if "peer_divergence" in r else "-"
            print(f"{clip[:31]:<32}{name:<20}{pdv:>9}{r['words']:>7}{r['ellipses']:>7}"
                  f"{r['rtf']:>6.2f}x{r['repeat_run']:>5}")
    print(f"\nTranscripts for side-by-side reading: {OUT_DIR}")


if __name__ == "__main__":
    main()
