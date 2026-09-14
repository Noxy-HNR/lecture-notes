"""Tune Cohere Transcribe for accuracy against HUMAN ground truth, under a realtime budget.

Why not tune on the user's own lectures: there's no verified transcript for them, and
tuning Cohere to agree more with Whisper/Qwen would partly tune it toward their mistakes.
So settings are chosen on public recordings with human transcripts, then sanity-checked on
the real lectures afterward.

  TED-LIUM long-form (lecture-like talks, 16kHz, full talks) - split so tuning can't overfit:
    dev  (tune on these):  6 talks, ~72 min
    test (confirm only):   4 different talks, ~76 min - never used to pick settings
  AMI (optional, --dataset ami): noisy spontaneous meeting speech, short utterances

Scoring matches the Open ASR Leaderboard: Whisper's EnglishTextNormalizer on both sides,
corpus-level WER via jiwer. Lower is better. RTF = processing seconds / audio seconds; the
live app needs RTF < 1.0.

What gets varied (defaults in parentheses = the app's current settings):
  mode     "hard": app-style fixed cuts every `chunk` seconds (hard, 20)
           "native": feed `window` seconds and let Cohere's own splitter cut at quiet points
                     into pieces up to `clip` seconds (as the model card intends for long audio)
  beams    beam search width (1 = greedy, the model's default)
  dither   feature-extractor dither (1e-5; deterministic, seeded by audio length)
  dtype    "bf16" or "fp32" (bf16)
  length_penalty (1.0), punctuation (True - irrelevant to normalized WER, kept for notes)

    python tools/cohere_tuning.py --stage A                 # chunking sweep on dev
    python tools/cohere_tuning.py --config '{"mode":"native","window":120,"beams":4}'
    python tools/cohere_tuning.py --split test --config ...  # confirm on held-out talks
Results accumulate in state/eval/cohere_tuning_results.json; finished runs are skipped.
"""
import argparse
import gc
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

COHERE_PATH = ROOT / "state" / "cohere_transcribe"
EVAL = ROOT / "state" / "eval"
TED = EVAL / "tedlium_long" / "data" / "test-00000-of-00001-7a1bb92f62e929b8.parquet"
AMI_DIR = EVAL / "ami" / "ami"
NORMALIZER = EVAL / "normalizer.json"
RESULTS = EVAL / "cohere_tuning_results.json"
HYPS = EVAL / "hyps"

DEV_TALKS = ["AimeeMullins", "DanBarber", "DanielKahneman", "EricMead_2009P_EricMead",
             "GaryFlake", "RobertGupta"]
TEST_TALKS = ["BillGates", "JamesCameron", "JaneMcGonigal", "MichaelSpecter"]

DEFAULTS = {"mode": "hard", "chunk": 20.0, "window": 120.0, "clip": 35.0, "beams": 1,
            "dither": 1e-5, "dtype": "bf16", "length_penalty": 1.0, "punctuation": True}

STAGES = {
    # Stage A: how audio is cut, all greedy. hard20 is exactly what the app does today.
    "A": [
        {"mode": "hard", "chunk": 20},
        {"mode": "hard", "chunk": 30},
        {"mode": "hard", "chunk": 35},
        {"mode": "native", "window": 60, "clip": 35},
        {"mode": "native", "window": 120, "clip": 35},
        {"mode": "native", "window": 300, "clip": 35},
        {"mode": "native", "window": 120, "clip": 30},
        {"mode": "native", "window": 120, "clip": 25},
    ],
}

MAX_BATCH = 4          # chunks decoded together; keeps beam search inside 12GB of VRAM
SAMPLE_RATE = 16000


def full_config(overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg


def config_key(cfg: dict) -> str:
    cut = (f"hard{cfg['chunk']:g}" if cfg["mode"] == "hard"
           else f"native{cfg['window']:g}-clip{cfg['clip']:g}")
    return (f"{cut}_b{cfg['beams']}_d{cfg['dither']:g}_{cfg['dtype']}"
            f"_lp{cfg['length_penalty']:g}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_ted(names: list[str], limit: int | None) -> list[dict]:
    table = pq.read_table(TED)
    items = []
    for i in range(table.num_rows):
        sid = table.column("speaker_id")[i].as_py()
        if sid not in names:
            continue
        samples, sr = sf.read(io.BytesIO(table.column("audio")[i].as_py()["bytes"]), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        assert sr == SAMPLE_RATE, f"{sid}: expected 16kHz, got {sr}"
        items.append({"id": sid, "audio": samples, "ref": table.column("text")[i].as_py()})
    items.sort(key=lambda x: names.index(x["id"]))
    return items[:limit] if limit else items


def load_ami(limit: int | None) -> list[dict]:
    items = []
    for path in sorted(AMI_DIR.glob("*.parquet")):
        table = pq.read_table(path)
        for i in range(table.num_rows):
            samples, sr = sf.read(io.BytesIO(table.column("audio")[i].as_py()["bytes"]), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                samples = librosa.resample(samples, orig_sr=sr, target_sr=SAMPLE_RATE)
            items.append({"id": table.column("id")[i].as_py(), "audio": samples,
                          "ref": table.column("text")[i].as_py()})
    return items[:limit] if limit else items


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_loaded = {}


def model_for(dtype: str):
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    if dtype in _loaded:
        return _loaded[dtype]
    _loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()
    proc = AutoProcessor.from_pretrained(str(COHERE_PATH))
    model = CohereAsrForConditionalGeneration.from_pretrained(
        str(COHERE_PATH), device_map="cuda",
        dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
    _loaded[dtype] = (proc, model)
    return proc, model


def transcribe_item(proc, model, cfg: dict, samples: np.ndarray) -> str:
    import torch
    fe = proc.feature_extractor
    fe.dither = cfg["dither"]
    if cfg["mode"] == "hard":
        step = int(cfg["chunk"] * SAMPLE_RATE)
        # Cohere's splitter cuts anything longer than (max_audio_clip_s - overlap_chunk_second).
        # For fixed cuts, raise the limit so each piece really is decoded whole, as the app does.
        fe.max_audio_clip_s = max(35.0, cfg["chunk"] + fe.overlap_chunk_second)
    else:
        step = int(cfg["window"] * SAMPLE_RATE)
        fe.max_audio_clip_s = cfg["clip"]
    max_new_tokens = max(440, int(fe.max_audio_clip_s * 22))

    texts = []
    for start in range(0, len(samples), step):
        window = samples[start:start + step]
        if len(window) < int(0.05 * SAMPLE_RATE):
            continue  # a few ms of tail isn't speech
        inputs = proc(window, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                      language="en", punctuation=cfg["punctuation"])
        inputs.pop("audio_chunk_index", None)  # one window = one sample, pieces stay in order
        n = inputs["input_features"].shape[0]
        for b in range(0, n, MAX_BATCH):
            batch = {}
            for k, v in inputs.items():
                if not hasattr(v, "shape"):
                    continue
                v = v[b:b + MAX_BATCH]
                batch[k] = (v.to(model.device, dtype=model.dtype) if v.is_floating_point()
                            else v.to(model.device))
            with torch.inference_mode():
                ids = model.generate(**batch, max_new_tokens=max_new_tokens,
                                     num_beams=cfg["beams"], length_penalty=cfg["length_penalty"])
            out = proc.decode(ids, skip_special_tokens=True)
            texts.extend(out if isinstance(out, list) else [out])
    return " ".join(t.strip() for t in texts if t and t.strip())


# ---------------------------------------------------------------------------
# Run + score
# ---------------------------------------------------------------------------

def load_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def save_results(results: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1), encoding="utf-8")
    tmp.replace(RESULTS)


def evaluate(split: str, items: list[dict], cfg: dict, normalizer) -> dict:
    import torch
    proc, model = model_for(cfg["dtype"])
    torch.cuda.reset_peak_memory_stats()
    refs, hyps, per_item = [], [], {}
    audio_s = sum(len(it["audio"]) for it in items) / SAMPLE_RATE
    started = time.time()
    for it in items:
        t0 = time.time()
        hyp = transcribe_item(proc, model, cfg, it["audio"])
        torch.cuda.synchronize()
        r, h = normalizer(it["ref"]), normalizer(hyp)
        refs.append(r)
        hyps.append(h)
        if split != "ami":
            per_item[it["id"]] = {"wer": jiwer.wer(r, h) if r.strip() else None,
                                  "seconds": round(time.time() - t0, 1)}
    elapsed = time.time() - started
    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    wer = jiwer.wer([r for r, _ in keep], [h for _, h in keep])
    HYPS.mkdir(parents=True, exist_ok=True)
    (HYPS / f"{split}__{config_key(cfg)}.json").write_text(
        json.dumps([{"id": it["id"], "hyp": h} for it, h in zip(items, hyps)], indent=1), encoding="utf-8")
    return {"split": split, "config": cfg, "wer": wer, "audio_s": audio_s,
            "seconds": elapsed, "rtf": elapsed / audio_s,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "per_item": per_item, "at": time.strftime("%Y-%m-%d %H:%M")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test", "ami"], default="dev")
    parser.add_argument("--stage", choices=list(STAGES))
    parser.add_argument("--config", action="append", default=[], help="JSON overrides; repeatable")
    parser.add_argument("--limit", type=int, help="first N talks/utterances only (smoke test)")
    parser.add_argument("--force", action="store_true", help="re-run configs already in results")
    parser.add_argument("--tag", default="", help="suffix for the results key, e.g. 'smoke'")
    args = parser.parse_args()

    from model_ab_test import recording_in_progress
    if recording_in_progress():
        print("A recording is live - not loading models onto the GPU. Stop the recorder first.")
        sys.exit(2)

    overrides = list(STAGES.get(args.stage, [])) + [json.loads(c) for c in args.config]
    if not overrides:
        parser.error("give --stage and/or --config")
    configs = [full_config(o) for o in overrides]

    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer(json.loads(NORMALIZER.read_text(encoding="utf-8")))

    items = (load_ami(args.limit) if args.split == "ami"
             else load_ted(DEV_TALKS if args.split == "dev" else TEST_TALKS, args.limit))
    audio_min = sum(len(it["audio"]) for it in items) / SAMPLE_RATE / 60
    print(f"{args.split}: {len(items)} items, {audio_min:.1f} min of audio | {len(configs)} config(s)", flush=True)

    results = load_results()
    for cfg in configs:
        key = f"{args.split}{('-' + args.tag) if args.tag else ''}|{config_key(cfg)}"
        if key in results and not args.force:
            r = results[key]
            print(f"  skip (done) {config_key(cfg):<42} WER {r['wer']:.2%}  RTF {r['rtf']:.3f}", flush=True)
            continue
        print(f"  running     {config_key(cfg):<42}", end=" ", flush=True)
        try:
            r = evaluate(args.split, items, cfg, normalizer)
        except Exception as e:
            print(f"FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            import torch
            torch.cuda.empty_cache()
            continue
        results[key] = r
        save_results(results)
        print(f"WER {r['wer']:.2%}  RTF {r['rtf']:.3f}  ({r['seconds']:.0f}s, {r['peak_vram_gb']:.1f}GB)", flush=True)

    prefix = f"{args.split}{('-' + args.tag) if args.tag else ''}|"
    rows = sorted((v for k, v in results.items() if k.startswith(prefix)), key=lambda v: v["wer"])
    if rows:
        base = next((v for v in rows if config_key(v["config"]) == config_key(full_config({}))), None)
        print(f"\n{args.split.upper()} RESULTS so far (lower WER is better; live app needs RTF < 1)")
        print(f"  {'config':<44}{'WER':>8}{'vs app':>9}{'RTF':>8}")
        for v in rows:
            delta = f"{(v['wer'] - base['wer']) * 100:+.2f}" if base else "-"
            print(f"  {config_key(v['config']):<44}{v['wer']:>7.2%}{delta:>9}{v['rtf']:>8.3f}")


if __name__ == "__main__":
    main()
