import sys
from types import SimpleNamespace

import numpy as np
import pytest

import audio
import capture
import main
import notes


def test_startup_validation_reads_existing_capture(monkeypatch):
    def forbidden(*args): pytest.fail('opened a second recording device')
    monkeypatch.setattr(audio, 'get_recorder', forbidden)
    existing = SimpleNamespace(startup_sample=lambda seconds: np.full(3 * 16000, .1, dtype=np.float32))
    blocks, message = main._check_audio_device('mic', existing)
    assert not blocks and 'OK' in message


def test_persisted_queue_reads_every_frame_from_wav(tmp_path):
    session = main.Session(tmp_path / 'queued.wav')
    first = session.persist_audio(np.full(16000, .1, dtype=np.float32))
    second = session.persist_audio(np.full(16000, .2, dtype=np.float32))
    assert isinstance(first, main.PersistedBlock)
    result = np.concatenate([first, second])
    session.close()
    assert len(result) == 32000
    np.testing.assert_array_equal(result[:16000], np.full(16000, .1, dtype=np.float32))
    np.testing.assert_array_equal(result[16000:], np.full(16000, .2, dtype=np.float32))


def test_startup_observation_keeps_queue_and_bounds_sample():
    device = capture.CaptureThread('mic')
    block = np.ones(8000, dtype=np.float32)
    for _ in range(20): device._observe_startup(block)
    assert device._startup_frames == 48000
    assert len(device.startup_sample(timeout=.01)) == 48000


def test_formatter_preserves_subscription_and_disables_tools(monkeypatch):
    monkeypatch.setattr(notes, '_find_claude_cli', lambda: 'claude')
    monkeypatch.delenv('NOTE_RECORDER_CLAUDE_BARE', raising=False)
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='Corrected transcript.', stderr='')
    monkeypatch.setattr(notes.subprocess, 'run', run)
    assert notes._try_claude_cli_format('source') == 'Corrected transcript.'
    args, kwargs = calls[0]
    assert '--safe-mode' in args and '--bare' not in args
    assert args[args.index('--tools') + 1] == ''
    assert '--no-session-persistence' in args
    assert kwargs['input'] == 'source'


@pytest.mark.parametrize('interrupt', [True, False])
def test_preparation_failure_or_interrupt_keeps_audio_and_closes_capture(tmp_path, monkeypatch, interrupt):
    import soundfile as sf
    actions = []
    class Preflight:
        def prepare_source(self, source): pass
        def start_audio_check(self, source): pass
        def start_gpu_warmup(self, mode): pass
        def finish(self, source, mode):
            assert sf.info(next(tmp_path.glob('*.wav'))).frames == 8000
            return False
    class Capture:
        def __init__(self, source, on_block=None): self.persist = on_block
        def start(self): self.persist(np.full(8000, .1, dtype=np.float32))
        def stop(self): actions.append('stopped')
        def abandon(self): actions.append('abandoned')
    monkeypatch.setattr(main, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(main, 'PreflightRunner', Preflight)
    monkeypatch.setattr(main.capture_module, 'CaptureThread', Capture)
    monkeypatch.setattr(main, 'choose_action', lambda args: 'record')
    monkeypatch.setattr(main, 'choose_class', lambda args: {'code': 'TEST 100', 'title': 'Test'})
    monkeypatch.setattr(notes, 'notes_path', lambda code: tmp_path / 'notes.md')
    monkeypatch.setattr(main.keep_awake, 'prevent_sleep', lambda: None)
    monkeypatch.setattr(main.keep_awake, 'allow_sleep', lambda: actions.append('sleep released'))
    monkeypatch.setattr(sys, 'argv', ['main.py', '--source', 'system', '--formatting', 'heuristic'])
    if interrupt:
        def interrupted(args): raise KeyboardInterrupt()
        monkeypatch.setattr(main, 'choose_formatting_mode', interrupted)
    with pytest.raises(KeyboardInterrupt if interrupt else SystemExit):
        main.run()
    assert actions == ['stopped', 'abandoned', 'sleep released']
    path = next(tmp_path.glob('*.wav'))
    with sf.SoundFile(path) as wav:
        assert len(wav.read()) == 8000
    assert not (tmp_path / 'notes.md').exists()
