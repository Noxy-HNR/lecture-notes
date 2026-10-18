"""Same-process ABBA encoder timing, with separate compilation costs and text parity."""
import argparse
import json
from pathlib import Path
import statistics
import time
from recorder_probe import ROOT, emit, isolate_performance, prepare_cuda, run_probe


def worker(args):
    isolate_performance()
    import torch
    prepare_cuda(torch)
    import transcribe
    import encoder_runtime
    import soundfile as sf
    backend = transcribe.get_model()
    modules = list(backend.model.encoder.layers)
    if args.mode == 'compile-fp8':
        modules += [linear for layer in backend.model.encoder.layers for ff in
                    (layer.feed_forward1, layer.feed_forward2) for linear in (ff.linear1, ff.linear2)]
    original = [(m, m.forward) for m in modules]
    stats = encoder_runtime.configure(backend.model, args.mode)
    compiled = [(m, m.forward) for m in modules]
    def select(label):
        for module, forward in original if label == 'eager' else compiled:
            module.forward = forward
    clips = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())
    short, sr = sf.read(clips[0]['path'], dtype='float32')
    cases = [('short', short)]
    for name, filename, start in [('continuous', 'BIOL_1440_20261008_125522.wav', 240),
                                ('quiet', 'PSYC_1300_20261007_130853.wav', 2160)]:
        with sf.SoundFile(ROOT / 'state' / filename) as source:
            source.seek(start * 16000)
            cases.append((name, source.read(120 * 16000, dtype='float32')))
    for name, samples in cases:
        first = {}
        reference = None
        for label in ['eager', 'compiled']:
            select(label)
            began = time.monotonic()
            result = backend.transcribe(samples, 16000)
            torch.cuda.synchronize()
            first[label] = time.monotonic() - began
            if label == 'eager': reference = result
        records = []
        for _ in range(3):
            for label in ['eager', 'compiled', 'compiled', 'eager']:
                select(label)
                began = time.monotonic()
                result = backend.transcribe(samples, 16000)
                torch.cuda.synchronize()
                records.append({'mode': label, 'seconds': time.monotonic() - began,
                                'segments_equal': result == reference})
        means = {label: statistics.mean(r['seconds'] for r in records if r['mode'] == label)
                 for label in ['eager', 'compiled']}
        emit({'case': name, 'first_seconds': first, 'means': means, 'runs': records, 'stats': stats})
    del original, compiled, modules, backend
    transcribe.unload_model()
    emit({'cuda_after_unload_gib': torch.cuda.memory_allocated() / 2**30})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['compile-bf16', 'compile-fp8'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child: worker(args)
    else: run_probe(Path(__file__).resolve(), ['--mode', args.mode, '--output', str(args.output)],
                    args.output, timeout=480, minimum_ram=4)
