"""Audio captured from the moment the app opens (capture.Preroll): what was said while the
menu was on screen ends up in the recording, in order and exactly once, and choosing anything
but Record drops it without writing it anywhere."""
import itertools
import threading
import time

import numpy as np
import pytest
import soundfile as sf

import capture
import main

SR = capture.audio.SAMPLE_RATE


class TickingRecorder:
    """Delivers 0.1 s blocks whose samples are all the block's number (1, 2, 3...), so the
    recording shows exactly which blocks reached it and in what order."""
    counters = {}

    def __init__(self, source, gate=None):
        self.source, self.gate = source, gate
        TickingRecorder.counters.setdefault(source, itertools.count(1))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def ticking(monkeypatch):
    TickingRecorder.counters = {}
    monkeypatch.setattr(capture.audio, "get_recorder", lambda source: TickingRecorder(source))

    def record_chunk(recorder, seconds):
        time.sleep(0.005)
        number = next(TickingRecorder.counters[recorder.source])
        return np.full(int(0.1 * SR), float(number), dtype=np.float32)

    monkeypatch.setattr(capture.audio, "record_chunk", record_chunk)


def make(source, on_block):
    return capture.CaptureThread(source, on_block=on_block)


def wait_for(condition, seconds=3.0):
    deadline = time.time() + seconds
    while time.time() < deadline and not condition():
        time.sleep(0.005)
    assert condition()


def block_numbers(samples):
    """The run of block numbers in a recording: one per 0.1 s."""
    return [int(v) for v in np.asarray(samples).reshape(-1, int(0.1 * SR))[:, 0]]


def test_adopted_audio_is_complete_in_order_and_gapless(ticking, tmp_path):
    preroll = capture.Preroll(["mic", "system"], make)
    wait_for(lambda: preroll.held_seconds("mic") >= 1.0)
    assert preroll.captures["mic"].audio_queue.empty(), "held audio is not queued for transcription yet"

    session = main.Session(tmp_path / "TEST_1000_20261115_090000.wav")
    mic, kept_seconds, recent = preroll.adopt("mic", session.persist_audio)
    assert kept_seconds >= 1.0 and recent is None  # under 3 s: the audio check samples fresh audio
    wait_for(lambda: session.total_frames >= int(2.5 * SR))  # the same capture keeps going
    mic.stop()
    session.close()

    recording, _ = sf.read(str(tmp_path / "TEST_1000_20261115_090000.wav"), dtype="float32")
    numbers = block_numbers(recording)
    assert numbers == list(range(1, len(numbers) + 1)), "every block from the app opening on, once, in order"

    queued = block_numbers(mic.drain_available())
    assert queued == numbers, "and all of it is queued for transcription from the durable recording"
    assert preroll.captures == {"mic": mic}, "the source that wasn't chosen was dropped"


def test_the_other_source_is_stopped_and_never_written(ticking, tmp_path):
    preroll = capture.Preroll(["mic", "system"], make)
    system = preroll.captures["system"]
    wait_for(lambda: preroll.held_seconds("system") > 0)
    session = main.Session(tmp_path / "TEST_1000_20261115_090000.wav")
    written = []
    mic, _, _ = preroll.adopt("mic", lambda block: written.append(float(block[0])) or session.persist_audio(block))
    system._thread.join(3)
    assert not system._thread.is_alive() and system.audio_queue.empty()
    mic.stop()
    session.close()
    assert written == sorted(written) and len(set(written)) == len(written)


def test_choosing_something_else_discards_everything(ticking):
    preroll = capture.Preroll(["mic", "system"], make)
    captures = list(preroll.captures.values())
    wait_for(lambda: preroll.held_seconds("mic") > 0)
    preroll.discard(everything=True)
    for thread in captures:
        thread._thread.join(3)
        assert not thread._thread.is_alive() and thread.audio_queue.empty()
    assert preroll.held_seconds("mic") == 0 and not preroll.has("mic")


def test_only_the_newest_audio_is_kept(ticking):
    preroll = capture.Preroll(["mic"], make, max_seconds=0.5)
    wait_for(lambda: preroll.held_seconds("mic") >= 0.5)
    time.sleep(0.3)  # dozens more blocks arrive; the total held must not grow
    assert 0.5 <= preroll.held_seconds("mic") <= 0.61
    kept = []
    mic, seconds, _ = preroll.adopt("mic", lambda block: kept.append(int(block[0])) or block)
    mic.stop()
    held = kept[:round(seconds * 10)]
    assert held == list(range(held[0], held[0] + len(held))) and held[0] > 1, "the oldest blocks were dropped"


def test_three_seconds_of_preroll_answers_the_audio_check_at_once(ticking):
    preroll = capture.Preroll(["mic"], make)
    wait_for(lambda: preroll.held_seconds("mic") >= 3.2, seconds=6)
    mic, _, recent = preroll.adopt("mic", lambda block: block)
    assert recent is not None and len(recent) >= 3 * SR
    mic.reset_startup_sample(recent)
    started = time.monotonic()
    sample = mic.startup_sample(timeout=0.5)
    assert time.monotonic() - started < 0.2 and len(sample) == 3 * SR
    np.testing.assert_array_equal(sample, recent[-3 * SR:])
    mic.reset_startup_sample(None)   # e.g. the microphone was muted until now: sample fresh audio
    assert not mic._startup_ready.is_set()
    mic.stop()


def test_blocks_arriving_during_the_hand_over_are_neither_lost_nor_doubled(monkeypatch):
    """The capture thread delivers a block while adopt() is still writing the held ones."""
    monkeypatch.setattr(capture.audio, "get_recorder", lambda source: TickingRecorder(source))
    preroll = capture.Preroll(["mic"], lambda source, on_block: capture.CaptureThread(source, on_block=on_block))
    preroll.captures["mic"].stop(timeout=0.1)  # this test delivers blocks by hand
    deliver = preroll.captures["mic"].on_block
    for number in (1, 2, 3):
        assert deliver(np.full(4, float(number), dtype=np.float32)) is capture.HOLD

    written, racing = [], []

    def persist(block):
        if not racing:  # first held block being written: a new block arrives right now
            racing.append(threading.Thread(target=lambda: written.append(("late", deliver(np.full(4, 4.0, dtype=np.float32))))))
            racing[0].start()
            time.sleep(0.05)
        written.append(float(block[0]))
        return block

    preroll.adopt("mic", persist)
    racing[0].join(2)
    assert [w for w in written if not isinstance(w, tuple)] == [1.0, 2.0, 3.0, 4.0]
    late = next(w for w in written if isinstance(w, tuple))[1]
    assert late is not capture.HOLD and float(late[0]) == 4.0


def test_watchdog_takes_over_a_running_capture_without_opening_another():
    class Running:
        started = 0

        def start(self):
            Running.started += 1

    existing = Running()
    made = []
    watchdog = main.CaptureWatchdog(lambda: made.append(Running()) or made[-1], existing=existing)
    assert watchdog.capture is existing and not made and Running.started == 0


def test_startup_timing_is_logged(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(main, "STARTUP_TIMING_LOG", tmp_path / "startup_timings.jsonl")
    monkeypatch.setattr(main.transcribe, "device_info", lambda: "cuda/bfloat16/nano/bulk")
    main._record_startup_timing({"ok": True, "waited_after_prompts": 1.5, "model": 4.2, "audio_check": 0.0})
    entry = json.loads((tmp_path / "startup_timings.jsonl").read_text(encoding="utf-8"))
    assert entry["model"] == 4.2 and entry["runtime"] == "cuda/bfloat16/nano/bulk" and "at" in entry
