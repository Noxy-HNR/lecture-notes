"""Short live A/B capture validation; keeps samples in memory and saves statistics.

    python tools/capture_diagnostics.py --source mic --seconds 6 --output report.json

Reports cannot establish speech recognition accuracy without spoken references.
Mono remains the recorder default. Stereo is downmixed only for comparison.
"""
import argparse
from pathlib import Path
import sys
import time

from recorder_probe import emit, isolate_performance, run_probe


def worker(args):
    isolate_performance()
    import audio
    import numpy as np
    device = (audio.pick_microphone() if args.source == "mic" else audio.pick_loopback_speaker())
    emit({"source": args.source, "device": device.name, "device_channels": device.channels,
          "sample_rate": audio.SAMPLE_RATE, "blocksize": audio.BLOCKSIZE,
          "limitation": "Sequential ambient samples; no speech reference or accuracy claim."})
    for channels in (1, 2, 2, 1):
        if device.channels < channels:
            emit({"channels": channels, "skipped": "Device has fewer channels"})
            continue
        gaps = [0]
        def count():
            gaps[0] += 1
        start = time.perf_counter()
        blocks = []
        with audio.count_discontinuities(count), audio.get_recorder(args.source, channels=channels) as recorder:
            for _ in range(int(args.seconds * 2)):
                blocks.append(recorder.record(numframes=audio.SAMPLE_RATE // 2))
        samples = np.concatenate(blocks).astype(np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        mono = samples.mean(axis=1)
        rms = np.sqrt(np.mean(samples.astype(np.float64)**2, axis=0))
        correlation = (float(np.corrcoef(samples.T)[0, 1]) if channels == 2 and np.all(rms > 0) else None)
        emit({"channels": channels, "frames": len(samples), "seconds": time.perf_counter() - start,
              "discontinuities": gaps[0], "finite": bool(np.isfinite(samples).all()),
              "rms_by_channel": rms.tolist(), "peak_by_channel": np.max(np.abs(samples), axis=0).tolist(),
              "dc_by_channel": np.mean(samples, axis=0).tolist(),
              "clipped_fraction": float(np.mean(np.abs(samples) >= .999)),
              "zero_fraction": float(np.mean(samples == 0)), "channel_correlation": correlation,
              "downmix_rms": float(np.sqrt(np.mean(mono.astype(np.float64)**2)))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["mic", "system"], default="mic")
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("Use 1-30 seconds per capture mode")
    if args.child:
        worker(args)
    else:
        run_probe(Path(__file__).resolve(), ["--source", args.source, "--seconds", str(args.seconds),
                  "--output", str(args.output)], args.output, timeout=4 * args.seconds + 45, minimum_ram=.5)
