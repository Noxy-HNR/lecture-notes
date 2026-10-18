"""Bounded real local alignment check against the accepted reference clip transcript."""
import argparse
import json
from pathlib import Path
import time
from recorder_probe import ROOT, isolate_performance, prepare_cuda, run_probe, emit


def worker(args):
    isolate_performance()
    import torch
    prepare_cuda(torch, required_free_gib=3)
    import alignment
    import soundfile as sf
    clips = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())
    began = time.monotonic()
    model = alignment.ForcedAligner()
    emit({'load_seconds': time.monotonic() - began})
    for clip in clips[:3]:
        samples, sr = sf.read(clip['path'], dtype='float32')
        segment = {'text': clip['reference'], 'start': 0, 'end': len(samples) / sr}
        began = time.monotonic()
        result = model.align(samples, segment)
        emit({'id': clip['id'], 'seconds': time.monotonic() - began, 'segment': result})
        assert result['text'] == segment['text']
    model.close()
    emit({'cuda_after_close_gib': torch.cuda.memory_allocated() / 2**30})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child: worker(args)
    else: run_probe(Path(__file__).resolve(), ['--output', str(args.output)], args.output, timeout=180, minimum_ram=3.5)
