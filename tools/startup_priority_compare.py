"""Fresh-process A-B-A startup comparison for the production and reference runtimes."""
import argparse
import json
import os
from pathlib import Path
import time
from recorder_probe import ROOT, isolate_performance, prepare_cuda, emit, run_probe


def worker(args):
    isolate_performance()
    os.environ['NOTE_RECORDER_ASR'] = args.runtime
    os.environ['NOTE_RECORDER_ENCODER'] = 'eager'
    start = time.monotonic()
    import transcribe
    dependencies = transcribe.start_dependency_preparation().result()
    imported = time.monotonic()
    prepare_cuda(dependencies[0])
    model = transcribe.get_model()
    ready = time.monotonic()
    import soundfile as sf
    clip = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())[0]
    samples, sr = sf.read(clip['path'], dtype='float32')
    result = model.transcribe(samples, sr)
    emit({'runtime': model.device_desc, 'dependency_seconds': imported - start,
          'model_ready_seconds': ready - start, 'first_text_seconds': time.monotonic() - start,
          'segments': result, 'loader': getattr(getattr(model, 'loader', None), 'stats', None)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', choices=['nano', 'hf'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child: worker(args)
    else:
        for index, runtime in enumerate(['nano', 'hf', 'nano']):
            output = args.output.with_name(args.output.stem + f'_{index}_{runtime}.json')
            run_probe(Path(__file__).resolve(), ['--runtime', runtime, '--output', str(output)],
                      output, timeout=120, minimum_ram=3.5)
