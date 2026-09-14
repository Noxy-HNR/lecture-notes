"""Measures transcription accuracy across chunk sizes, to pick the size empirically.

Same method as vad_ab_test.py: transcribe the clip once with no artificial chunk
boundaries at all and treat that as the reference, then measure how far each chunk size
diverges from it. Lower WER = that chunk size is damaging the audio less at its seams.

The hypothesis worth testing is that the current 15s is context-starving Whisper: the
pause-aware experiment's shorter chunks (mean 12-13s) scored measurably WORSE than 15s
on two lectures, which suggests accuracy is still climbing with chunk length and the
optimum may be past where we are.

    python tools/chunk_size_sweep.py <wav> [<wav> ...] [--seconds 300] [--sizes 15 20 25 30]
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


def sweep_clip(wav: str, seconds: float, sizes: list[float]) -> dict:
    with sf.SoundFile(wav) as f:
        samples = f.read(int(seconds * audio.SAMPLE_RATE), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    name = Path(wav).name
    print(f"\n{'=' * 70}\n{name}  ({len(samples) / audio.SAMPLE_RATE:.0f}s)\n{'=' * 70}")

    print("Reference (single pass, no chunk boundaries)...")
    t0 = time.time()
    reference = " ".join(s["text"] for s in transcribe.transcribe_chunk_segments(samples)).strip()
    print(f"  {len(reference.split())} words in {time.time() - t0:.1f}s")

    results = {}
    for size in sizes:
        chunks = chunk_fixed(samples, size)
        t0 = time.time()
        text = transcribe_chunks(chunks)
        elapsed = time.time() - t0
        wer = word_error_rate(reference, text)
        results[size] = {"wer": wer, "seconds": elapsed, "words": len(text.split()),
                          "chunks": len(chunks)}
        print(f"  {size:>4.0f}s chunks ({len(chunks):>2} chunks): "
              f"WER {wer:>6.2%}  |  {elapsed:>5.1f}s  |  {len(text.split())} words")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wavs", nargs="+", help="One or more real lecture .wav files")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--sizes", type=float, nargs="+", default=[15, 20, 25, 30])
    args = parser.parse_args()

    transcribe.use_backend("whisper")  # tunes Whisper; don't silently measure the default Cohere model
    transcribe.get_model()
    all_results = {Path(w).name: sweep_clip(w, args.seconds, args.sizes) for w in args.wavs}

    print(f"\n{'=' * 70}\nSUMMARY (WER vs single-pass reference - lower is better)\n{'=' * 70}")
    header = "clip".ljust(34) + "".join(f"{s:>9.0f}s" for s in args.sizes)
    print(header)
    for name, results in all_results.items():
        row = name[:33].ljust(34)
        best = min(results, key=lambda s: results[s]["wer"])
        for size in args.sizes:
            mark = "*" if size == best else " "
            row += f"{results[size]['wer']:>8.2%}{mark}"
        print(row)

    # Average across clips so one unusually hard lecture doesn't decide it alone.
    print("\naverage".ljust(34) + "".join(
        f"{sum(r[s]['wer'] for r in all_results.values()) / len(all_results):>8.2%} "
        for s in args.sizes))
    avg = {s: sum(r[s]["wer"] for r in all_results.values()) / len(all_results) for s in args.sizes}
    best = min(avg, key=avg.get)
    current = avg.get(15.0)
    print(f"\nBest average: {best:.0f}s chunks at {avg[best]:.2%} WER"
          + (f" (current 15s: {current:.2%}, delta {avg[best] - current:+.2%})"
             if current is not None else ""))
    print("* = best for that clip")


if __name__ == "__main__":
    main()
