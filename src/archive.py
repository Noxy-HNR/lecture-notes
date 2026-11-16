"""Compresses finished lecture recordings and ages out very old ones.

A lecture is recorded as a float32 16 kHz mono .wav (about 230 MB an hour) because the
recorder appends to it while transcribing. Once the lecture is over, the .wav's job is
recovery (--resume re-transcribes it), playback from search and highlights, and the
corrections page. Ogg Opus at about 31 kbit/s does that at roughly a fifteenth of the
size, and speech stays clear. (Re-transcribing from the compressed copy has not been
compared against the original; the transcript made from the original is always kept.)

    compress_finished()  - every .wav older than COMPRESS_AFTER_HOURS -> .ogg, then the
                           .wav is deleted, but only after the .ogg was read back whole
    prune_old()          - removes compressed recordings older than RETENTION_DAYS
                           (transcripts are always kept)

Both skip the recording in progress, and never raise: they run as housekeeping.
"""
import os
import time
from pathlib import Path

import numpy as np

STATE_DIR = Path(__file__).resolve().parent.parent / "state"

COMPRESS_AFTER_HOURS = 24      # the day of the lecture is when --resume actually gets used
RETENTION_DAYS = 180           # a compressed hour is ~15 MB, so a semester's worth is cheap
BLOCK_FRAMES = 16000 * 30      # 30 s at a time, so a long lecture never sits in RAM whole
LENGTH_TOLERANCE_SECONDS = 1.0  # Opus pads the start and end by a few milliseconds
OPUS_COMPRESSION = 0.9         # libsndfile level 0..1, not linear: 0.9 is ~31 kbit/s for this audio (1.0 drops to ~6)


def _active_recording() -> str | None:
    """The stem of the recording in progress, if any (a fresh heartbeat in status.json)."""
    try:
        import telemetry
        status = telemetry.read_status()
    except Exception:
        return None
    if not status.get("active") or time.time() - float(status.get("updated_at") or 0) > 120:
        return None
    return Path(status.get("wav_path") or "").stem or None


def compress(wav: Path) -> Path:
    """Writes `<stem>.ogg` next to the .wav, checks it reads back at the same length, then
    deletes the .wav. Raises on any failure, leaving the .wav untouched."""
    import soundfile as sf
    ogg = wav.with_suffix(".ogg")
    partial = wav.with_name(wav.stem + ".ogg.partial")
    with sf.SoundFile(str(wav)) as source:
        rate, channels, frames = source.samplerate, source.channels, source.frames
        with sf.SoundFile(str(partial), "w", samplerate=rate, channels=channels,
                          format="OGG", subtype="OPUS", compression_level=OPUS_COMPRESSION) as out:
            while True:
                block = source.read(BLOCK_FRAMES, dtype="float32", always_2d=False)
                if not len(block):
                    break
                out.write(np.clip(block, -1.0, 1.0))
    written = 0
    with sf.SoundFile(str(partial)) as check:
        if check.samplerate != rate:
            raise ValueError(f"compressed copy has sample rate {check.samplerate}, expected {rate}")
        while True:
            block = check.read(BLOCK_FRAMES, dtype="float32", always_2d=False)
            if not len(block):
                break
            written += len(block)
    if abs(written - frames) > LENGTH_TOLERANCE_SECONDS * rate:
        raise ValueError(f"compressed copy is {written / rate:.0f}s long, the recording {frames / rate:.0f}s")
    stat = wav.stat()
    os.replace(partial, ogg)
    os.utime(ogg, (stat.st_atime, stat.st_mtime))  # retention counts from the lecture, not from today
    wav.unlink()
    return ogg


def compress_finished(now: float | None = None, log=print) -> dict:
    """Compresses every finished recording. Returns {"compressed", "failed", "saved_bytes"}."""
    now = time.time() if now is None else now
    result = {"compressed": 0, "failed": 0, "saved_bytes": 0}
    try:
        candidates = sorted(STATE_DIR.glob("*.wav"))
    except Exception:
        return result
    for wav in candidates:
        try:
            if _active_recording():
                break  # a lecture is recording; encoding waits so it never competes with transcription
            if wav.stem.endswith("_clip"):
                continue
            if now - wav.stat().st_mtime < COMPRESS_AFTER_HOURS * 3600:
                continue
            before = wav.stat().st_size
            ogg = compress(wav)
            result["compressed"] += 1
            result["saved_bytes"] += before - ogg.stat().st_size
            log(f"Compressed {wav.name}: {before / 1e6:.0f} MB -> {ogg.stat().st_size / 1e6:.1f} MB")
        except Exception as error:
            result["failed"] += 1
            wav.with_name(wav.stem + ".ogg.partial").unlink(missing_ok=True)
            log(f"Could not compress {wav.name} ({type(error).__name__}: {error}); the .wav is kept")
    return result


def prune_old(now: float | None = None, log=print) -> int:
    """Deletes compressed recordings older than RETENTION_DAYS. Returns how many."""
    now = time.time() if now is None else now
    removed = 0
    try:
        old = [p for p in STATE_DIR.glob("*.ogg") if now - p.stat().st_mtime > RETENTION_DAYS * 86400]
    except Exception:
        return 0
    for path in old:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        log(f"Removed {removed} compressed recording(s) older than {RETENTION_DAYS} days (transcripts kept)")
    return removed
