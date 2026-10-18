"""Bounded HF/Nano quality and speed comparison with human reference transcripts.

Each backend runs alone in a fresh child process. Validation streams four held-out
TED-LIUM talks plus cached AMI utterances; no authentic lecture is sent to a cloud.
Use --backend production for the integrated runtime, or nano for the pinned
upstream experiment. Use --dataset smoke before the larger validation.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

from recorder_probe import ROOT, emit, isolate_performance, prepare_cuda, run_probe


def items(args):
    import soundfile as sf
    import pyarrow.parquet as pq
    if args.dataset == "lectures":
        for filename, start in [("PSYC_1300_20261007_130853.wav", 2160),
                                ("BIOL_1440_20261008_125522.wav", 240)]:
            with sf.SoundFile(ROOT / "state" / filename) as source:
                source.seek(start * source.samplerate)
                samples = source.read(120 * source.samplerate, dtype="float32")
                if source.samplerate != 16000 or source.channels != 1:
                    raise ValueError("Expected mono 16 kHz lecture audio")
            yield "unreferenced_lecture", f"{filename}@{start}", samples, None
        return
    if args.dataset == "smoke":
        manifest = json.loads((ROOT / "state/startup_experiment/clips.json").read_text(encoding="utf-8"))
        for row in manifest:
            samples, sr = sf.read(row["path"], dtype="float32")
            yield "reference", row["id"], samples, row["reference"]
        return
    names = {"BillGates", "JamesCameron", "JaneMcGonigal", "MichaelSpecter"}
    ted = ROOT / "state/eval/tedlium_long/data/test-00000-of-00001-7a1bb92f62e929b8.parquet"
    for batch in pq.ParquetFile(ted).iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        if row["speaker_id"] not in names:
            continue
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != 16000 or samples.ndim != 1:
            raise ValueError("Expected mono 16 kHz TED audio")
        yield "ted", row["speaker_id"], samples, row["text"]
    count = 0
    ami = ROOT / "state/eval/ami/ami/test-00004-of-00015.parquet"
    for batch in pq.ParquetFile(ami).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if sr != 16000 or samples.ndim != 1:
                raise ValueError("Expected mono 16 kHz AMI audio")
            yield "ami", row["id"], samples, row["text"]
            count += 1
            if count >= args.ami_limit:
                return


def worker(args):
    started = time.perf_counter()
    isolate_performance()
    import torch
    import transcribe
    prepare_cuda(torch)
    if args.backend in ('hf', 'production'):
        import os
        os.environ['NOTE_RECORDER_ASR'] = 'hf' if args.backend == 'hf' else 'nano'
    if args.backend in ('hf', 'production'):
        dependencies = transcribe.start_dependency_preparation().result()
    else:
        sys.path.insert(0, str(ROOT / "state/efficiency_experiment/nano"))
        import nano_cohere_transcribe
    imported = time.perf_counter()
    if args.backend in ('hf', 'production'):
        backend = transcribe.get_model()
    else:
        from nano_experiment import load_backend
        backend = load_backend(ROOT, prompt=args.prompt, max_graphs=args.max_graphs,
                               graph_bytes=args.graph_buffer_mib * 2**20, graphs=not args.no_graphs)
    torch.cuda.synchronize()
    emit({"phase": "ready", "backend": args.backend, "prompt": args.prompt,
          "graphs": args.backend == "nano" and not args.no_graphs,
          "dependency_seconds": imported - started, "weight_seconds": time.perf_counter() - imported,
          "ready_seconds": time.perf_counter() - started})
    import soundfile as sf
    import jiwer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    normalize = EnglishTextNormalizer(json.loads((ROOT / "state/eval/normalizer.json").read_text()))
    # Record cold versus warm cost explicitly, with the actual 770-token graph key.
    clip = json.loads((ROOT / "state/startup_experiment/clips.json").read_text())[0]
    samples, sr = sf.read(clip["path"], dtype="float32")
    for phase in ("first_decode", "warm_decode"):
        begin = time.perf_counter()
        segments = backend.transcribe(samples, sr)
        torch.cuda.synchronize()
        emit({"phase": phase, "seconds": time.perf_counter() - begin, "segments": segments})
    refs, hyps, duration, elapsed = {}, {}, {}, {}
    for dataset, identifier, samples, reference in items(args):
        begin = time.perf_counter()
        parts, lengths, timed_segments = [], [], []
        for start in range(0, len(samples), 120 * 16000):
            window = samples[start:start + 120 * 16000]
            lengths.extend(backend._piece_lengths(window))
            segments = backend.transcribe(window, 16000)
            parts.extend(s["text"] for s in segments)
            if reference is None:
                timed_segments.extend({**s, "start": s["start"] + start / 16000,
                                       "end": s["end"] + start / 16000} for s in segments)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        hypothesis = " ".join(parts)
        error_rate = None
        if reference is not None:
            r, h = normalize(reference), normalize(hypothesis)
            error_rate = jiwer.wer(r, h)
            refs.setdefault(dataset, []).append(r)
            hyps.setdefault(dataset, []).append(h)
            duration[dataset] = duration.get(dataset, 0) + len(samples) / 16000
            elapsed[dataset] = elapsed.get(dataset, 0) + seconds
        emit({"dataset": dataset, "id": identifier, "reference": reference, "hypothesis": hypothesis,
              "wer": error_rate, "seconds": seconds, "audio_seconds": len(samples) / 16000,
              "piece_samples": lengths, **({"segments": timed_segments} if reference is None else {})})
    for dataset in refs:
        emit({"phase": "summary", "dataset": dataset, "items": len(refs[dataset]),
              "corpus_wer": jiwer.wer(refs[dataset], hyps[dataset]), "seconds": elapsed[dataset],
              "audio_seconds": duration[dataset], "rtf": elapsed[dataset] / duration[dataset],
              "reference_words": sum(len(r.split()) for r in refs[dataset])})
    cache = getattr(backend, "graph_cache", None)
    emit({"phase": "memory", "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
          "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
          "encoder": getattr(backend, 'encoder_stats', None),
          "graph_cache": ({k: getattr(cache, k) for k in ["peak_bytes", "peak_entries", "evictions", "builds"]}
                          if cache else None)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "nano", "production"], required=True)
    parser.add_argument("--dataset", choices=["smoke", "validation", "lectures"], default="smoke")
    parser.add_argument("--prompt", choices=["hf", "native"], default="hf")
    parser.add_argument("--ami-limit", type=int, default=200)
    parser.add_argument("--max-graphs", type=int, default=2)
    parser.add_argument("--no-graphs", action="store_true", help="Nano ablation: same model/frontend with eager decoding")
    parser.add_argument("--graph-buffer-mib", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ami_limit < 1 or args.max_graphs < 1 or args.graph_buffer_mib < 1:
        parser.error("Dataset and graph limits must be positive")
    if args.child:
        worker(args)
    else:
        arguments = sys.argv[1:]
        run_probe(Path(__file__).resolve(), arguments, args.output, timeout=args.timeout)
