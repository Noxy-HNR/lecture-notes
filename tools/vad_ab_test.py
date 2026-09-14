"""Offline A/B test: does pause-aware chunking transcribe more accurately than fixed-size?

Rationale for the method: there is no human ground-truth transcript for these lectures,
so instead we transcribe the clip in ONE pass with no artificial chunk boundaries at all
and treat that as the reference. That isolates exactly the variable under test - a
single-pass transcription has no boundary artifacts by construction, so whichever
chunking strategy diverges from it LESS is the one damaging the audio less at its seams.

Run:
    python tools/vad_ab_test.py <wav> [--seconds 300] [--chunk 15]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import soundfile as sf

import audio
import capture as capture_module
import transcribe


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER: Levenshtein distance over word sequences / reference length."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0
    # Classic DP edit-distance table, one row at a time to keep memory small.
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current[j] = min(previous[j] + 1,        # deletion
                              current[j - 1] + 1,     # insertion
                              previous[j - 1] + cost)  # substitution
        previous = current
    return previous[-1] / len(ref)


def _blocks(samples: np.ndarray, block_seconds: float = capture_module.READ_SECONDS):
    """Splits audio into the same size blocks the live capture thread produces, so the
    boundary logic sees exactly the granularity it would see in a real session."""
    step = int(block_seconds * audio.SAMPLE_RATE)
    return [samples[i:i + step] for i in range(0, len(samples), step)]


def chunk_fixed(samples: np.ndarray, chunk_seconds: float) -> list[np.ndarray]:
    step = int(chunk_seconds * audio.SAMPLE_RATE)
    return [samples[i:i + step] for i in range(0, len(samples), step)]


def chunk_pause_aware(samples: np.ndarray, chunk_seconds: float) -> list[np.ndarray]:
    """Mirrors CaptureThread.collect_chunk()'s accumulate-and-test loop, calling the same
    capture.ends_on_pause() the live path uses."""
    max_frames = int(chunk_seconds * audio.SAMPLE_RATE)
    min_frames = int(min(capture_module.MIN_CHUNK_SECONDS, chunk_seconds) * audio.SAMPLE_RATE)
    tail_frames = int(capture_module.PAUSE_TAIL_SECONDS * audio.SAMPLE_RATE)

    chunks, accumulating, total = [], [], 0
    for block in _blocks(samples):
        accumulating.append(block)
        total += len(block)
        hit_cap = total >= max_frames
        on_pause = (total >= min_frames
                    and capture_module.ends_on_pause(accumulating, tail_frames))
        if hit_cap or on_pause:
            chunks.append(np.concatenate(accumulating))
            accumulating, total = [], 0
    if accumulating:
        chunks.append(np.concatenate(accumulating))
    return chunks


def transcribe_chunks(chunks: list[np.ndarray], vocab_prompt=None) -> str:
    """Feeds chunks through the real RollingTranscriber, so the overlap window and
    rolling-context prompt behave exactly as they do live."""
    import main  # imported lazily - pulls in the whole app, only needed here
    transcriber = main.RollingTranscriber(vocab_prompt)
    out = []
    for chunk in chunks:
        for seg in transcriber.process(chunk):
            out.append(seg["text"])
    return " ".join(out).strip()


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Path to a real lecture .wav from state/")
    parser.add_argument("--seconds", type=float, default=300, help="How much of the clip to use")
    parser.add_argument("--chunk", type=float, default=15.0, help="Max chunk size (upper bound)")
    args = parser.parse_args()

    with sf.SoundFile(args.wav) as f:
        samples = f.read(int(args.seconds * audio.SAMPLE_RATE), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    print(f"Clip: {args.wav} ({len(samples) / audio.SAMPLE_RATE:.0f}s)\n")

    # This tunes Whisper's chunking. Pin it: the app's default model is now Cohere, which
    # would otherwise be measured here silently.
    transcribe.use_backend("whisper")
    transcribe.get_model()  # load once so timings below exclude model load

    print("Reference (single pass, no chunk boundaries)...")
    t0 = time.time()
    reference = " ".join(s["text"] for s in transcribe.transcribe_chunk_segments(samples)).strip()
    print(f"  {len(reference.split())} words in {time.time() - t0:.1f}s\n")

    results = {}
    for label, chunks in (("fixed", chunk_fixed(samples, args.chunk)),
                           ("pause-aware", chunk_pause_aware(samples, args.chunk))):
        sizes = [len(c) / audio.SAMPLE_RATE for c in chunks]
        print(f"{label}: {len(chunks)} chunks "
              f"(min {min(sizes):.1f}s / mean {sum(sizes)/len(sizes):.1f}s / max {max(sizes):.1f}s)")
        t0 = time.time()
        text = transcribe_chunks(chunks)
        elapsed = time.time() - t0
        wer = word_error_rate(reference, text)
        results[label] = (wer, elapsed, len(text.split()))
        print(f"  {len(text.split())} words in {elapsed:.1f}s -> WER vs reference: {wer:.2%}\n")

    print("=" * 60)
    for label, (wer, elapsed, words) in results.items():
        print(f"{label:>12}: WER {wer:.2%}  |  {elapsed:.1f}s  |  {words} words")
    better = min(results, key=lambda k: results[k][0])
    delta = abs(results["fixed"][0] - results["pause-aware"][0])
    print(f"\nLower divergence from reference: {better} (by {delta:.2%} WER)")


if __name__ == "__main__":
    main_cli()
