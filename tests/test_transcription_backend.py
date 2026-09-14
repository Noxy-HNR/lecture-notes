"""Tests for the transcription backend switch and the vocabulary auto-learning shutoff.

No GPU, model files or network: backends are replaced with fakes, so these run in
milliseconds and test the wiring rather than a model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

import audio
import transcribe


class FakeBackend:
    def __init__(self, name="fake", timestamps=True):
        self.name = name
        self.timestamps = timestamps
        self.device_desc = "fake/float16"
        self.calls = []

    def transcribe(self, samples, sample_rate, initial_prompt=None):
        seconds = len(samples) / sample_rate
        self.calls.append(seconds)
        return [{"text": f"chunk {len(self.calls)}", "start": 0.0, "end": seconds}]


@pytest.fixture(autouse=True)
def reset_backend():
    transcribe.use_backend(None)
    yield
    transcribe.use_backend(None)


def _speech(seconds):
    return np.full(int(seconds * audio.SAMPLE_RATE), 0.1, dtype=np.float32)


class TestBackendSelection:
    def test_prefers_cohere_when_it_loads(self, monkeypatch):
        monkeypatch.setitem(transcribe._LOADERS, "cohere", lambda: FakeBackend("cohere-transcribe", False))
        monkeypatch.setitem(transcribe._LOADERS, "whisper", lambda: FakeBackend("whisper-large-v3"))
        assert transcribe.backend_name() == "cohere-transcribe"
        assert transcribe.fallback_reason() == ""

    def test_falls_back_to_whisper_and_says_why(self, monkeypatch):
        """A deleted or broken default model must never stop a recording from starting -
        which is what happened when a cache cleanup removed the Whisper weights mid-week."""
        def broken():
            raise FileNotFoundError("model files not found")
        monkeypatch.setitem(transcribe._LOADERS, "cohere", broken)
        monkeypatch.setitem(transcribe._LOADERS, "whisper", lambda: FakeBackend("whisper-large-v3"))
        assert transcribe.backend_name() == "whisper-large-v3"
        assert "cohere" in transcribe.fallback_reason()
        assert "model files not found" in transcribe.fallback_reason()

    def test_clear_error_when_nothing_loads(self, monkeypatch):
        def broken():
            raise RuntimeError("nope")
        monkeypatch.setitem(transcribe._LOADERS, "cohere", broken)
        monkeypatch.setitem(transcribe._LOADERS, "whisper", broken)
        with pytest.raises(RuntimeError, match="no transcription model could load"):
            transcribe.get_model()

    def test_use_backend_pins_a_single_model(self, monkeypatch):
        """tools/ that tune Whisper settings must not silently measure Cohere instead."""
        monkeypatch.setitem(transcribe._LOADERS, "cohere", lambda: FakeBackend("cohere-transcribe", False))
        monkeypatch.setitem(transcribe._LOADERS, "whisper", lambda: FakeBackend("whisper-large-v3"))
        transcribe.use_backend("whisper")
        assert transcribe.backend_name() == "whisper-large-v3"

    def test_timestampless_backend_returns_one_segment_spanning_the_chunk(self, monkeypatch):
        monkeypatch.setitem(transcribe._LOADERS, "cohere", lambda: FakeBackend("cohere-transcribe", False))
        segments = transcribe.transcribe_chunk_segments(_speech(20.0))
        assert len(segments) == 1 and segments[0]["start"] == 0.0 and segments[0]["end"] == 20.0


class TestRollingOverlap:
    def _run_two_chunks(self, monkeypatch, timestamps):
        import main
        fake = FakeBackend(timestamps=timestamps)
        monkeypatch.setitem(transcribe._LOADERS, "cohere", lambda: fake)
        rolling = main.RollingTranscriber(vocab_prompt=None)
        rolling.process(_speech(20.0))
        rolling.process(_speech(20.0))
        return fake.calls

    def test_no_overlap_without_timestamps(self, monkeypatch):
        """Without timestamps there's no way to drop the words that came from the repeated
        1.5s, so it would duplicate speech at every boundary."""
        assert self._run_two_chunks(monkeypatch, timestamps=False) == [20.0, 20.0]

    def test_overlap_kept_for_timestamped_backends(self, monkeypatch):
        calls = self._run_two_chunks(monkeypatch, timestamps=True)
        assert calls[0] == 20.0
        assert calls[1] == pytest.approx(20.0 + main_overlap())


class TestLongWindows:
    def test_pieces_get_their_own_times(self):
        """A 120s window comes back as ~35s pieces; each needs its real place in the window,
        and an empty piece must still advance the clock."""
        sr = audio.SAMPLE_RATE
        segs = transcribe.segments_from_pieces(["one", "", "three"], [30 * sr, 35 * sr, 20 * sr], sr)
        assert [(s["text"], s["start"], s["end"]) for s in segs] == [
            ("one", 0.0, 30.0), ("three", 65.0, 85.0)]

    def test_chunk_defaults_to_the_models_window(self, monkeypatch):
        import argparse
        import main
        fake = FakeBackend("cohere-transcribe", False)
        fake.preferred_chunk_seconds = 120.0
        monkeypatch.setitem(transcribe._LOADERS, "cohere", lambda: fake)
        assert main.resolve_chunk_seconds(argparse.Namespace(chunk=None)) == 120.0
        assert main.resolve_chunk_seconds(argparse.Namespace(chunk=15.0)) == 15.0

    def test_brief_speech_in_a_long_silent_window_is_still_transcribed(self, monkeypatch):
        """The whole-window RMS of 10s of speech in 120s of quiet falls under the silence
        threshold; the gate must look at slices, not the average."""
        import main
        fake = FakeBackend("cohere-transcribe", False)
        monkeypatch.setitem(transcribe._LOADERS, "cohere", lambda: fake)
        window = np.zeros(120 * audio.SAMPLE_RATE, dtype=np.float32)
        window[50 * audio.SAMPLE_RATE:60 * audio.SAMPLE_RATE] = 0.005
        assert audio.is_silent(window)  # the old whole-window check would have skipped it
        rolling = main.RollingTranscriber(vocab_prompt=None)
        assert rolling.process(window) and rolling.consecutive_silent_chunks == 0
        assert rolling.process(np.zeros(120 * audio.SAMPLE_RATE, dtype=np.float32)) == []


class TestCohereFailureGuards:
    """Two failures from CHEM 1450 on 2026-09-14: invented sentences for digital silence, and
    a greedy-decoding loop on very quiet room audio. The normal decode is faked; the guard is real."""
    SR = audio.SAMPLE_RATE

    def _backend(self, window_texts, piece_seconds, repair_texts=()):
        backend = object.__new__(transcribe._CohereBackend)
        backend.decode_calls = []
        repairs = list(repair_texts)

        def decode(samples, sample_rate):
            backend.decode_calls.append(round(len(samples) / sample_rate, 2))
            return list(window_texts) if len(backend.decode_calls) == 1 else [repairs.pop(0)]

        backend._decode = decode
        backend._piece_lengths = lambda samples: [int(s * self.SR) for s in piece_seconds]
        return backend

    def test_digital_silence_gets_no_text_and_normal_pieces_are_untouched(self):
        window = np.concatenate([_speech(30.0), np.zeros(30 * self.SR, dtype=np.float32)])
        backend = self._backend(["Carbon tends to be black.",
                                 "The world is a very important part of the world. " * 6], [30, 30])
        segments = backend.transcribe(window, self.SR)
        assert [(s["text"], s["start"], s["end"]) for s in segments] == [("Carbon tends to be black.", 0.0, 30.0)]
        assert backend.decode_calls == [60.0]  # nothing re-decoded

    def test_looping_piece_is_redecoded_in_sub_pieces(self):
        quiet_room = np.full(30 * self.SR, 0.01, dtype=np.float32)  # quiet, but above the silence gate
        window = np.concatenate([quiet_room, _speech(20.0)])
        backend = self._backend(["the other one is " * 190, "Nitrogen is blue."], [30, 20],
                                repair_texts=["Find this question.", "", "I'm going to go to the next one."])
        segments = backend.transcribe(window, self.SR)
        assert [(s["text"], s["start"], s["end"]) for s in segments] == [
            ("Find this question. I'm going to go to the next one.", 0.0, 30.0),
            ("Nitrogen is blue.", 30.0, 50.0)]
        assert backend.decode_calls == [50.0, 10.0, 10.0, 10.0]
        assert backend.repaired_pieces == 1

    def test_sub_piece_that_still_loops_is_dropped(self):
        backend = self._backend(["the other one is " * 190], [20],
                                repair_texts=["the other one is " * 12, "Okay."])
        segments = backend.transcribe(np.full(20 * self.SR, 0.01, dtype=np.float32), self.SR)
        assert [s["text"] for s in segments] == ["Okay."]

    @pytest.mark.parametrize("text, seconds, expected", [
        ("Thank you. Thank you. Thank you.", 5, False),      # TED's longest repeat: 3
        ("I'm going to go to the next slide. " * 3, 20, False),
        ("No, no, no, no, no. That's the wrong carbon.", 5, False),
        ("Step 1 is this, step 2 is this, step 3 is this, step 4 is this, step 5 is this.", 12, False),
        ("Carbon 1, carbon 2, carbon 3, carbon 4, carbon 5, carbon 6.", 8, False),
        ("the other one is " * 4, 20, True),
        ("I've never seen a film before. " * 4, 30, True),
        ("bang " * 10, 10, True),
        ("word " * 770, 33, True),                            # 23 words/second
        (" ".join(f"word{i % 97} and{i % 89}" for i in range(60)), 33, False),  # 120 words in 33s, no repeats
    ])
    def test_degenerate_thresholds(self, text, seconds, expected):
        assert transcribe.looks_degenerate(text, seconds) is expected

    def test_silence_check_looks_at_slices(self):
        mostly_zero = np.zeros(30 * self.SR, dtype=np.float32)
        mostly_zero[10 * self.SR:12 * self.SR] = 0.1  # 2s of sound inside 30s of zeros
        assert transcribe.is_silent_audio(np.zeros(30 * self.SR, dtype=np.float32), self.SR)
        assert not transcribe.is_silent_audio(mostly_zero, self.SR)


def main_overlap():
    import main
    return main.RollingTranscriber.OVERLAP_SECONDS


class TestVocabAutoLearningOff:
    def test_proofreading_does_not_write_vocab(self, tmp_path, monkeypatch):
        """Auto-learned terms ("R-isomer", "antiemetic") leaked back into the prompt and
        were transcribed in place of words actually said. Proofreading must not grow
        vocab.json any more, even when its correction looks like a learnable term."""
        import notes
        import vocab
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text('{"BIOL 1440": ["osmosis"]}\n', encoding="utf-8")
        monkeypatch.setattr(vocab, "VOCAB_PATH", vocab_file)
        monkeypatch.setattr(notes, "_try_claude_cli_format",
                            lambda prompt: "the mitochondria produce energy")
        monkeypatch.setattr(notes, "_try_claude_api_format", lambda prompt: None)

        before = vocab_file.read_text(encoding="utf-8")
        result = notes._proofread("Biology", "BIOL 1440", "the mitocondria produce energy")

        assert result == "the mitochondria produce energy"
        assert vocab_file.read_text(encoding="utf-8") == before
