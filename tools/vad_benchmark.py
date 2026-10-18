"""Conservative whole-piece VAD evaluation; never trims or concatenates speech.

Uses a pinned Silero ONNX model under ignored state/efficiency_experiment/silero.
Scans human-referenced noisy AMI speech and quiet/noisy variants of short reference
clips. Candidate thresholds are declared before evaluation; no recorder defaults change.
"""
import argparse
import io
import json
from pathlib import Path
import time

from recorder_probe import ROOT, emit, isolate_performance, prepare_cuda, run_probe


class SileroProbabilities:
    """16 kHz CPU-only ONNX inference, matching upstream state/context handling."""
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])

    def __call__(self, samples):
        import numpy as np
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        probabilities = []
        for start in range(0, len(samples), 512):
            block = np.asarray(samples[start:start + 512], dtype=np.float32)
            block = np.pad(block, (0, 512 - len(block)))[None, :]
            joined = np.concatenate([context, block], axis=1)
            out, state = self.session.run(None, {"input": joined, "state": state,
                                                "sr": np.array(16000, dtype=np.int64)})
            context = joined[:, -64:]
            probabilities.append(float(out[0, 0]))
        return probabilities


def worker(args):
    isolate_performance()
    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq
    import transcribe
    vad = SileroProbabilities(args.model)
    thresholds = [.1, .2, .5]
    speech = []
    started = time.perf_counter()
    parquet = ROOT / "state/eval/ami/ami/test-00004-of-00015.parquet"
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if sr != 16000:
                raise ValueError("Expected cached 16 kHz AMI")
            if samples.ndim != 1:
                samples = samples.mean(axis=1)
            probabilities = vad(samples)
            speech.append({"id": row["id"], "reference": row["text"], "seconds": len(samples) / sr,
                           "max_probability": max(probabilities, default=0),
                           "rms_skipped": transcribe.is_silent_audio(samples, sr)})
            if len(speech) >= args.ami_limit:
                break
        if len(speech) >= args.ami_limit:
            break
    emit({"ami_speech": speech, "seconds": time.perf_counter() - started,
          "thresholds": thresholds,
          "false_rejections": {str(t): sum(r["max_probability"] < t for r in speech if r["reference"].strip())
                               for t in thresholds},
          "rms_false_rejections": sum(r["rms_skipped"] for r in speech if r["reference"].strip())})
    # Explicitly synthetic amplitude/noise stress cases; do not present as real quiet-class WER.
    manifest = json.loads((ROOT / "state/startup_experiment/clips.json").read_text(encoding="utf-8"))
    cases = []
    rng = np.random.default_rng(20261011)
    for clip in manifest:
        samples, sr = sf.read(clip["path"], dtype="float32")
        for name, changed in [("original", samples), ("quiet_30db", samples * .0316228),
                              ("noise_5db_snr", samples + rng.normal(0, np.sqrt(np.mean(samples**2)) / 10**(.25),
                                                                   len(samples)).astype(np.float32))]:
            probabilities = vad(changed)
            cases.append({"id": clip["id"], "condition": name, "reference": clip["reference"],
                          "samples": changed, "max_probability": max(probabilities, default=0),
                          "rms_skipped": transcribe.is_silent_audio(changed, sr)})
    negatives = []
    for name, samples in [("digital_zero", np.zeros(30 * 16000, dtype=np.float32)),
                          ("white_noise", rng.normal(0, .01, 30 * 16000).astype(np.float32)),
                          ("60hz_hum", (.01 * np.sin(np.arange(30 * 16000) * 2 * np.pi * 60 / 16000)).astype(np.float32))]:
        probabilities = vad(samples)
        negatives.append({"id": name, "max_probability": max(probabilities, default=0),
                          "rms_skipped": transcribe.is_silent_audio(samples, 16000)})
    emit({"synthetic_negative_controls": negatives})
    import torch
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    import jiwer
    prepare_cuda(torch)
    normalizer = EnglishTextNormalizer(json.loads((ROOT / "state/eval/normalizer.json").read_text()))
    backend = transcribe.get_model()
    grouped = {}
    for case in cases:
        # Decode once without the RMS guard to expose what either gate would discard.
        start = time.perf_counter()
        text = " ".join(backend._decode(case["samples"], 16000))
        seconds = time.perf_counter() - start
        record = {k: v for k, v in case.items() if k != "samples"}
        record.update(hypothesis=text, decode_seconds=seconds)
        grouped.setdefault(case["condition"], []).append(record)
    for condition, records in grouped.items():
        refs = [normalizer(r["reference"]) for r in records]
        scores = {"ungated": jiwer.wer(refs, [normalizer(r["hypothesis"]) for r in records]),
                  "rms": jiwer.wer(refs, ["" if r["rms_skipped"] else normalizer(r["hypothesis"]) for r in records])}
        for threshold in thresholds:
            scores[f"vad_{threshold}"] = jiwer.wer(refs, [normalizer(r["hypothesis"]) if r["max_probability"] >= threshold
                                                         else "" for r in records])
        emit({"condition": condition, "corpus_wer": scores, "cases": records,
              "reference_words": sum(len(r.split()) for r in refs)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "state/efficiency_experiment/silero/silero_vad.onnx")
    parser.add_argument("--ami-limit", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ami_limit < 1:
        parser.error("AMI limit must be positive")
    if args.child:
        worker(args)
    else:
        run_probe(Path(__file__).resolve(), ["--model", str(args.model), "--ami-limit", str(args.ami_limit),
                  "--output", str(args.output)], args.output, timeout=240)
