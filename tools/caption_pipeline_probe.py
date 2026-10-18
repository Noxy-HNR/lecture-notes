"""Exercise the real caption subprocess, packet queue, callbacks and owned cleanup."""
import argparse
import json
from pathlib import Path
import time
from recorder_probe import ROOT, emit, isolate_performance, run_probe


def worker(args):
    isolate_performance()
    import streaming
    import soundfile as sf
    clip = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())[0]
    samples, sr = sf.read(clip['path'], dtype='float32')
    events = []
    captions = streaming.Captions(events.append, terms=['interject', 'appropriate'])
    try:
        assert captions.ready.wait(20), 'caption worker did not become ready'
        for start in range(0, len(samples), 8000):
            captions.submit(samples[start:start + 8000])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and len([e for e in events if e['type'] == 'caption']) < 6:
            time.sleep(.05)
        assert any(e.get('text') for e in events)
    finally:
        captions.close()
    emit({'events': events, 'returncode': captions.process.poll(),
          'thread_alive': captions.thread.is_alive(), 'dropped': captions.dropped})
    assert captions.process.poll() is not None and not captions.thread.is_alive()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child: worker(args)
    else: run_probe(Path(__file__).resolve(), ['--output', str(args.output)], args.output, timeout=90)
