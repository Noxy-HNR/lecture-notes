"""Offline comparisons. Refuses to run while a recorder window is open.

accuracy: JSON list of {audio, reference, start?, seconds?, quiet_padding?}.
References must be human-checked, not another model's output.
format: compare separate/combined prompts without changing notes or using local GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

def wer(reference, hypothesis):
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    row = list(range(len(hyp)+1))
    for i, word in enumerate(ref, 1):
        nxt = [i]
        for j, other in enumerate(hyp, 1):
            nxt.append(min(row[j]+1, nxt[-1]+1, row[j-1]+(word != other)))
        row = nxt
    return row[-1] / max(1, len(ref))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['accuracy', 'format'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--class-code', default='CHEM')
    parser.add_argument('--class-title', default='Chemistry')
    args = parser.parse_args()
    import recorder_launcher
    if recorder_launcher.recorder_state()['running']:
        parser.error('Recorder is open; close it normally after class before benchmarking.')
    if args.output.exists():
        parser.error('Choose a new output path; existing reports are preserved.')
    results = []
    if args.mode == 'format':
        import notes
        text = args.input.read_text(encoding='utf-8')
        for combined in (False, True):
            start = time.perf_counter()
            if combined:
                prompt = notes._build_combined_prompt(args.class_title, args.class_code, 'Benchmark', text, '')
            else:
                proof = notes._try_claude_cli_format(notes._build_proofread_prompt(args.class_title, args.class_code, text))
                if proof is None:
                    raise RuntimeError('Proofreading failed; no fallback substituted in comparison')
                prompt = notes._build_notes_prompt(args.class_title, args.class_code, 'Benchmark', proof, '')
            output = notes._try_claude_cli_format(prompt)
            if output is None:
                raise RuntimeError('Formatting failed')
            results.append({'combined': combined, 'seconds': time.perf_counter()-start,
                            'output': output, 'review_required': 'Compare omissions, factual changes, uncertainty flags, and duplicates.'})
    else:
        import numpy as np
        import soundfile as sf
        import audio
        import main as recorder
        cases = json.loads(args.input.read_text(encoding='utf-8'))
        for case in cases:
            path = args.input.parent / case['audio']
            with sf.SoundFile(path) as stream:
                if stream.samplerate != audio.SAMPLE_RATE or stream.channels != 1:
                    raise ValueError('Use mono 16 kHz reference audio')
                stream.seek(int(case.get('start', 0)*audio.SAMPLE_RATE))
                samples = stream.read(int(case.get('seconds', 120)*audio.SAMPLE_RATE), dtype='float32')
            padding = int(case.get('quiet_padding', 0)*audio.SAMPLE_RATE)
            samples = np.pad(samples, (padding, padding))
            start = time.perf_counter()
            segments = recorder.RollingTranscriber().process(samples)
            hypothesis = ' '.join(s['text'] for s in segments)
            results.append({'case': case, 'hypothesis': hypothesis, 'segments': segments,
                            'wer': wer(case['reference'], hypothesis), 'seconds': time.perf_counter()-start})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
