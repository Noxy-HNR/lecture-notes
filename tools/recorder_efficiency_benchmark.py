"""Compare predecode silence filtering using the same loaded weights and features.

Requires a working-tree transcribe.py snapshot from before the change. ABBA order
reduces warm-cache bias. Does not record audio or change notes/recorder telemetry.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from recorder_probe import ROOT, emit, isolate_performance, prepare_cuda, run_probe


def worker(args):
    isolate_performance()
    import soundfile as sf
    import transcribe
    import torch
    prepare_cuda(torch)
    backend = transcribe.get_model()
    spec = importlib.util.spec_from_file_location("baseline_transcribe", args.baseline / "transcribe.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    baseline = object.__new__(old._CohereBackend)
    baseline.__dict__.update(backend.__dict__)
    # Initialize kernels with reference speech before either timed candidate.
    manifest = json.loads((ROOT / "state/startup_experiment/clips.json").read_text(encoding="utf-8"))
    samples, sr = sf.read(manifest[0]["path"], dtype="float32")
    backend.transcribe(samples, sr)
    cases = [("mixed_quiet", "PSYC_1300_20261007_130853.wav", 2160),
             ("continuous_speech", "BIOL_1440_20261008_125522.wav", 240)]
    for name, filename, start in cases:
        with sf.SoundFile(ROOT / "state" / filename) as f:
            f.seek(start * f.samplerate)
            samples = f.read(120 * f.samplerate, dtype="float32")
            sr = f.samplerate
        records = []
        for label, model in [("baseline", baseline), ("filtered", backend),
                             ("filtered", backend), ("baseline", baseline)]:
            before = model.generated_tokens
            torch.cuda.synchronize()
            started = time.perf_counter()
            segments = model.transcribe(samples, sr)
            torch.cuda.synchronize()
            records.append({"candidate": label, "seconds": time.perf_counter() - started,
                            "tokens": model.generated_tokens - before, "segments": segments})
        equal = all(r["segments"] == records[0]["segments"] for r in records)
        emit({"case": name, "segments_equal": equal,
              "piece_seconds": [n / sr for n in backend._piece_lengths(samples)], "runs": records})
    # Human-reference clips also verify ordinary speech parity.
    parity = []
    for clip in manifest:
        samples, sr = sf.read(clip["path"], dtype="float32")
        a, b = baseline.transcribe(samples, sr), backend.transcribe(samples, sr)
        parity.append({"id": clip["id"], "segments_equal": a == b})
    emit({"reference_clip_parity": parity, "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "state/efficiency_experiment/baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        worker(args)
    else:
        run_probe(Path(__file__).resolve(), ["--baseline", str(args.baseline), "--output", str(args.output)], args.output)
