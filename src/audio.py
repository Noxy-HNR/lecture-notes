"""Audio capture: microphone or system-audio (WASAPI loopback) via the `soundcard` library."""
import warnings
import threading
from contextlib import contextmanager

import numpy as np
import soundcard as sc

SAMPLE_RATE = 16000  # what Cohere Transcribe wants
BLOCKSIZE = 4096
_capture_warnings = threading.local()


class _SoundcardWarnings:
    """Route just WASAPI's discontinuity warning to the current capture's counter.

    A catch_warnings context changes global filters on Python 3.13. This proxy is
    local to SoundCard's Windows backend and uses thread-local ownership instead.
    Other warnings, and captures without a counter, retain normal warning behavior.
    """
    def __init__(self, original, category):
        self.original, self.category = original, category

    def __getattr__(self, name):
        return getattr(self.original, name)

    def warn(self, message, category=None, stacklevel=1, source=None):
        counter = getattr(_capture_warnings, "counter", None)
        if (str(message) == "data discontinuity in recording"
                and category is self.category and counter is not None):
            counter()
            return
        self.original.warn(message, category, stacklevel=stacklevel + 1, source=source)


if hasattr(sc, "mediafoundation"):
    _wasapi = sc.mediafoundation
    if not isinstance(_wasapi.warnings, _SoundcardWarnings):
        _wasapi.warnings = _SoundcardWarnings(_wasapi.warnings, _wasapi.SoundcardRuntimeWarning)


@contextmanager
def count_discontinuities(counter):
    previous = getattr(_capture_warnings, "counter", None)
    _capture_warnings.counter = counter
    try:
        yield
    finally:
        _capture_warnings.counter = previous


def pick_microphone():
    return sc.default_microphone()


def pick_loopback_speaker():
    """Default speaker, opened for loopback recording of whatever is playing."""
    return sc.default_speaker()


def get_recorder(source: str, *, channels: int = 1):
    """
    source: "mic" or "system"
    Returns an object with a `.record(numframes)` method via a context-manager-friendly wrapper.
    """
    if source == "mic":
        mic = pick_microphone()
        return mic.recorder(samplerate=SAMPLE_RATE, channels=channels, blocksize=BLOCKSIZE)
    elif source == "system":
        speaker = pick_loopback_speaker()
        loopback_mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        return loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=channels, blocksize=BLOCKSIZE)
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
    Speech models hallucinate filler or whole sentences when fed silence - this catches a muted mic, a suspended/disconnected audio device, or
    dead air on the "system audio" source before it ever reaches the model, rather than
    relying on VAD alone (which doesn't always fully suppress a whole silent chunk)."""
    if chunk_audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(chunk_audio, dtype=np.float64))))
    return rms < threshold
