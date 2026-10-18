"""Verify true native streaming in isolation using public reference speech."""
import argparse
import json
from pathlib import Path
import time
from recorder_probe import ROOT, isolate_performance, emit, run_probe


def worker(args):
    isolate_performance()
    from native_speech import Recognizer
    import soundfile as sf
    began = time.monotonic()
    model = Recognizer()
    emit({'load_seconds': time.monotonic() - began})
    clips = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())
    for clip in clips[:3]:
        model.reset()
        samples, sr = sf.read(clip['path'], dtype='float32')
        events = []
        began = time.monotonic()
        for start in range(0, len(samples), 8000):
            events.extend(model.push(samples[start:start + 8000]))
        events.extend(model.finish())
        emit({'id': clip['id'], 'seconds': time.monotonic() - began, 'events': events,
              'reference': clip['reference']})
        assert any(e['text'] for e in events)
    model.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child: worker(args)
    else: run_probe(Path(__file__).resolve(), ['--output', str(args.output)], args.output, timeout=180)
