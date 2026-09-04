"""Does speaker diarization earn what it costs?

It runs on the whole session at Ctrl+C, adds real shutdown latency, and pulls in ~2GB of
dependencies (torch + pyannote). It has never actually been validated - the original
complaint about this feature was "I don't notice the speaker diarization doing anything".

This measures what it produces on real lecture audio rather than assuming:
  - how long it takes, relative to the audio length
  - how many speakers it finds (a lecture should be one dominant voice plus occasional
    students; finding one speaker means no Q&A can ever be labelled, and finding a dozen
    means it's splitting the lecturer up)
  - how lopsided the talk-time is (the instructor should dominate)
  - how much of the transcript would actually get a non-instructor label - i.e. how much
    Q&A content the feature is really adding

    python tools/diarization_check.py <wav> [--seconds 600]
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf

import audio
import diarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--seconds", type=float, default=600)
    args = parser.parse_args()

    if not diarize.available():
        print("Diarization is not configured (no HUGGINGFACE_TOKEN) - nothing to measure.")
        return

    src = Path(args.wav)
    with sf.SoundFile(str(src)) as f:
        samples = f.read(int(args.seconds * audio.SAMPLE_RATE), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    clip_seconds = len(samples) / audio.SAMPLE_RATE

    clip = src.with_name(src.stem + "_diarcheck.wav")
    sf.write(str(clip), samples, audio.SAMPLE_RATE, subtype="FLOAT")
    try:
        print(f"{src.name}: {clip_seconds:.0f}s of audio")
        print("Running diarization (first run also downloads/loads the model)...")
        t0 = time.time()
        segments = diarize.diarize(clip)
        elapsed = time.time() - t0

        if not segments:
            print(f"  FAILED - returned nothing after {elapsed:.1f}s")
            return

        talk = Counter()
        for s in segments:
            talk[s["speaker"]] += s["end"] - s["start"]
        total = sum(talk.values()) or 1.0
        ranked = talk.most_common()

        print(f"\n  took {elapsed:.1f}s for {clip_seconds:.0f}s of audio "
              f"({elapsed / clip_seconds:.2f}x realtime)")
        print(f"  {len(segments)} turns across {len(talk)} speaker(s)\n")
        for speaker, seconds in ranked:
            share = seconds / total
            role = "instructor (most talk time)" if speaker == ranked[0][0] else "other"
            print(f"    {speaker:<12} {seconds:>7.1f}s  {share:>6.1%}  {role}")

        non_primary = 1 - (ranked[0][1] / total)
        print(f"\n  Talk time that is NOT the primary speaker: {non_primary:.1%}")
        if len(talk) == 1:
            print("  -> Only one speaker found: no Q&A can ever be labelled from this.")
        elif non_primary < 0.02:
            print("  -> Under 2% non-instructor speech: the Q&A feature has almost "
                  "nothing to work with on this lecture.")
        else:
            print("  -> There is real non-instructor speech here for Q&A extraction "
                  "to pick up.")
    finally:
        clip.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
