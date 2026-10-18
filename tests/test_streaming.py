import queue
import threading
from types import SimpleNamespace

import numpy as np

from streaming import Captions
from streaming_worker import read_exact
import streaming


def test_caption_backlog_resets_state_and_keeps_latest_audio():
    captions = object.__new__(Captions)
    captions.queue = queue.Queue(maxsize=2)
    captions.closed = threading.Event()
    captions.dropped = 0
    block = np.ones(8000, dtype=np.float32)
    captions.submit(block)
    captions.submit(block)
    captions.submit(block * 2)
    assert captions.dropped == 1
    assert captions.queue.get_nowait() is None
    np.testing.assert_array_equal(np.frombuffer(captions.queue.get_nowait(), dtype='<f4'), block * 2)


def test_short_pipe_reads_reassemble_packet():
    chunks = iter([b'a', b'bc', b'd'])
    assert read_exact(SimpleNamespace(read=lambda n: next(chunks)), 4) == b'abcd'


def test_worker_exit_during_startup_reports_failure_and_releases_job(monkeypatch):
    import gpu_formatter
    import health
    events, closed = [], []
    process = SimpleNamespace(stdout=[], stderr=[], poll=lambda: 7)
    monkeypatch.setattr(streaming.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(gpu_formatter, '_attach_windows_kill_job', lambda p: 'owned job')
    monkeypatch.setattr(gpu_formatter, '_close_windows_job', closed.append)
    monkeypatch.setattr(health, '_read_memory', lambda: (4 * 2**30, 16 * 2**30))
    worker = Captions(events.append)
    worker.thread.join(timeout=2)
    assert not worker.thread.is_alive()
    assert worker.closed.is_set()
    assert any(event['type'] == 'error' and 'before becoming ready' in event['message']
               for event in events)
    assert closed == ['owned job']
