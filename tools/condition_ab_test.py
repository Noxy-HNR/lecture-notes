"""A/B: does letting Whisper condition on its own previous text help or hurt here?

transcribe.CONDITION_ON_PREVIOUS_TEXT is currently False, which prevents a bad
transcription from propagating forward and compounding - Whisper's classic failure mode
is looping or hallucinating once it goes off the rails. But it also denies the model
cross-chunk context, and this workload has already been shown to be context-hungry:
raising the chunk size from 15s to 20s cut WER by 3.75 points. So the setting is worth
testing rather than assuming.

Same method as the other harnesses: WER against a single-pass reference. The reference
is transcribed under each setting separately, since the setting affects it too.

    python tools/condition_ab_test.py <wav> [<wav> ...] [--seconds 300]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

import audio
import transcribe
from vad_ab_test import chunk_fixed, transcribe_chunks, word_error_rate


def run_clip(wav: str, seconds: float, chunk_seconds: float) -> dict:
    with sf.SoundFile(wav) as f:
        samples = f.read(int(seconds * audio.SAMPLE_RATE), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    print(f"\n{'=' * 70}\n{Path(wav).name}  ({len(samples) / audio.SAMPLE_RATE:.0f}s)\n{'=' * 70}")
    results = {}
    for setting in (False, True):
        transcribe.CONDITION_ON_PREVIOUS_TEXT = setting
        label = f"condition_on_previous_text={setting}"

        t0 = time.time()
        reference = " ".join(s["text"] for s in
                              transcribe.transcribe_chunk_segments(samples)).strip()
        ref_time = time.time() - t0

        chunks = chunk_fixed(samples, chunk_seconds)
        t0 = time.time()
        text = transcribe_chunks(chunks)
        elapsed = time.time() - t0
        wer = word_error_rate(reference, text)
        results[setting] = {"wer": wer, "seconds": elapsed, "words": len(text.split()),
                             "ref_words": len(reference.split())}
        print(f"  {label:<38} WER {wer:>6.2%}  |  {elapsed:>5.1f}s  |  "
              f"{len(text.split())} words (ref {len(reference.split())}, {ref_time:.0f}s)")
    transcribe.CONDITION_ON_PREVIOUS_TEXT = False  # restore the shipped default
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wavs", nargs="+")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--chunk", type=float, default=20.0)
    args = parser.parse_args()

    transcribe.get_model()
    all_results = {Path(w).name: run_clip(w, args.seconds, args.chunk) for w in args.wavs}

    print(f"\n{'=' * 70}\nSUMMARY (WER vs single-pass reference under the same setting)\n{'=' * 70}")
    print("clip".ljust(34) + "condition=False".rjust(17) + "condition=True".rjust(17))
    for name, r in all_results.items():
        print(name[:33].ljust(34) + f"{r[False]['wer']:>16.2%} " + f"{r[True]['wer']:>16.2%}")
    avg_off = sum(r[False]["wer"] for r in all_results.values()) / len(all_results)
    avg_on = sum(r[True]["wer"] for r in all_results.values()) / len(all_results)
    print("average".ljust(34) + f"{avg_off:>16.2%} " + f"{avg_on:>16.2%}")
    print(f"\nconditioning on previous text is "
          f"{'BETTER' if avg_on < avg_off else 'WORSE'} by {abs(avg_on - avg_off):.2%} WER")


if __name__ == "__main__":
    main()
