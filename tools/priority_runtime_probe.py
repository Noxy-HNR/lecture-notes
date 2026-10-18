"""Exercise the production runtime in a bounded offline child, without recording."""
import argparse
import json
from pathlib import Path
import time
from recorder_probe import ROOT, emit, isolate_performance, prepare_cuda, run_probe


def worker(args):
    isolate_performance()
    started = time.perf_counter()
    import torch
    prepare_cuda(torch)
    import transcribe
    model = transcribe.get_model()
    emit({'ready_seconds': time.perf_counter() - started, 'runtime': model.device_desc,
          'loader': getattr(getattr(model, 'loader', None), 'stats', None)})
    import soundfile as sf
    clips = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())
    for clip in clips:
        samples, sr = sf.read(clip['path'], dtype='float32')
        began = time.perf_counter()
        segments = model.transcribe(samples, sr)
        torch.cuda.synchronize()
        emit({'id': clip['id'], 'segments': segments, 'seconds': time.perf_counter() - began})
    for filename, start in [('BIOL_1440_20261008_125522.wav', 240),
                            ('PSYC_1300_20261007_130853.wav', 2160)]:
        with sf.SoundFile(ROOT / 'state' / filename) as source:
            source.seek(start * source.samplerate)
            samples = source.read(120 * source.samplerate, dtype='float32')
            sr = source.samplerate
        began = time.perf_counter()
        segments = model.transcribe(samples, sr)
        torch.cuda.synchronize()
        emit({'id': filename, 'segments': segments, 'seconds': time.perf_counter() - began})
    emit({'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30,
          'graph_builds': getattr(model, 'graph_cache', None).builds if hasattr(model, 'graph_cache') else 0})
    del model
    transcribe.unload_model()
    emit({'after_unload_cuda_gib': torch.cuda.memory_allocated() / 2**30})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        worker(args)
    else:
        run_probe(Path(__file__).resolve(), ['--output', str(args.output)], args.output, timeout=180)
