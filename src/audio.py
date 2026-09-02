"""Audio capture: microphone or system-audio (WASAPI loopback) via the `soundcard` library."""
import warnings

import numpy as np
import soundcard as sc

SAMPLE_RATE = 16000  # what faster-whisper wants
BLOCKSIZE = 4096      # smaller WASAPI buffer periods -> fewer discontinuity warnings

# soundcard/WASAPI occasionally reports "data discontinuity" (a few dropped ms of audio)
# when the system briefly can't keep up with the capture buffer. It's a known, mostly
# harmless quirk for speech transcription (loses a syllable at worst, not the recording) -
# silence it so it doesn't spam the console.
warnings.filterwarnings("ignore", message="data discontinuity in recording")


def pick_microphone():
    return sc.default_microphone()


def pick_loopback_speaker():
    """Default speaker, opened for loopback recording of whatever is playing."""
    return sc.default_speaker()


def get_recorder(source: str):
    """
    source: "mic" or "system"
    Returns an object with a `.record(numframes)` method via a context-manager-friendly wrapper.
    """
    if source == "mic":
        mic = pick_microphone()
        return mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCKSIZE)
    elif source == "system":
        speaker = pick_loopback_speaker()
        loopback_mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        return loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCKSIZE)
    else:
        raise ValueError(f"Unknown audio source: {source}")


def record_chunk(recorder, seconds: float) -> np.ndarray:
    """Record `seconds` of mono float32 audio at SAMPLE_RATE, returns 1-D numpy array."""
    numframes = int(seconds * SAMPLE_RATE)
    data = recorder.record(numframes=numframes)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


SILENCE_RMS_THRESHOLD = 0.002  # below this, treat the chunk as silence, don't transcribe it


def is_silent(chunk_audio: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """True if `chunk_audio` is at/near total silence (RMS amplitude below threshold).
    Whisper hallucinates repeated punctuation/filler ("...", "you", "thank you") when
    fed silence - this catches a muted mic, a suspended/disconnected audio device, or
    dead air on the "system audio" source before it ever reaches the model, rather than
    relying on VAD alone (which doesn't always fully suppress a whole silent chunk)."""
    if chunk_audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(chunk_audio, dtype=np.float64))))
    return rms < threshold
