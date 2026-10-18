"""Bounded local vocabulary-conditioned decoding and acceptance-gate smoke test."""
import argparse
import json
from pathlib import Path
import time
from recorder_probe import ROOT, emit, isolate_performance, prepare_cuda, run_probe


def worker(args):
    isolate_performance()
    import torch
    prepare_cuda(torch, required_free_gib=3)
    import vocabulary_retry as retry
    import soundfile as sf
    clip = json.loads((ROOT / 'state/startup_experiment/clips.json').read_text())[0]
    samples, sr = sf.read(clip['path'], dtype='float32')
    began = time.monotonic()
    model = retry.VocabularyASR()
    first = model.transcribe(samples)
    second = model.transcribe(samples, ['interject', 'appropriate'])
    emit({'id': clip['id'], 'seconds': time.monotonic() - began, 'unprompted': first,
          'contextual': second, 'reference': clip['reference']})
    if 'interject' in first:
        # Artificial misspelling tests wiring; it is not a claim of improved real ASR WER.
        synthetic = first.replace('interject', 'interjet', 1)
        selection = retry.target(synthetic, ['interject'])
        corrected = retry.accepted_text(synthetic, selection, first, second)
        emit({'synthetic_near_miss': synthetic, 'accepted': corrected,
              'paired_agreement': retry.normalized(first) == retry.normalized(second)})
    model.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child: worker(args)
    else: run_probe(Path(__file__).resolve(), ['--output', str(args.output)], args.output, timeout=180, minimum_ram=3.5)
