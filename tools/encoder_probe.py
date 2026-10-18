"""Compile one encoder candidate under a timeout and RAM watchdog, then compare output."""
import argparse
import json
from pathlib import Path
import time
from recorder_probe import ROOT, emit, isolate_performance, prepare_cuda, run_probe


def worker(args):
    isolate_performance()
    import torch
    prepare_cuda(torch)
    import transcribe
    import encoder_runtime
    backend = transcribe.get_model()
    import soundfile as sf
    clips = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())
    baseline = []
    for clip in clips:
        samples, sr = sf.read(clip['path'], dtype='float32')
        baseline.append(backend.transcribe(samples, sr))
    stats = encoder_runtime.configure(backend.model, args.mode)
    for clip, reference in zip(clips, baseline):
        samples, sr = sf.read(clip['path'], dtype='float32')
        for phase in ['first', 'warm']:
            begin = time.monotonic()
            result = backend.transcribe(samples, sr)
            torch.cuda.synchronize()
            emit({'id': clip['id'], 'phase': phase, 'seconds': time.monotonic() - begin,
                  'segments_equal': result == reference, 'stats': stats})
    emit({'mode': args.mode, 'stats': stats, 'cuda_peak_gib': torch.cuda.max_memory_allocated() / 2**30})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['compile-bf16', 'compile-fp8'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child: worker(args)
    else: run_probe(Path(__file__).resolve(), ['--mode', args.mode, '--output', str(args.output)],
                    args.output, timeout=480, minimum_ram=4)
