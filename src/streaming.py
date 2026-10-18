"""Provisional local captions. Capture never waits for this optional subprocess."""
from collections import deque
import json
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time

import native_speech


def available():
    return native_speech.available()


class Captions:
    def __init__(self, on_event, terms=()):
        self.on_event = on_event
        self.queue = queue.Queue(maxsize=32)
        self.closed = threading.Event()
        self.ready = threading.Event()
        self.dropped = 0
        self.process = self.job = None
        self.error_tail = deque(maxlen=20)
        self.terms = list(terms)[:64]
        self.thread = threading.Thread(target=self._run, daemon=True, name='caption-writer')
        self.thread.start()

    def submit(self, samples):
        if self.closed.is_set(): return
        import numpy as np
        packet = np.asarray(samples, dtype='<f4').tobytes()
        if len(packet) > 16000 * 4:
            raise ValueError('Caption packet exceeds one second')
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.dropped += 1
            # Native state must reset across a dropped interval. The WAV remains complete.
            while True:
                try: self.queue.get_nowait()
                except queue.Empty: break
            self.queue.put_nowait(None)
            self.queue.put_nowait(packet)

    def _events(self):
        try:
            for line in self.process.stdout:
                event = json.loads(line)
                if event['type'] == 'ready': self.ready.set()
                self.on_event(event)
        except Exception as error:
            self.on_event({'type': 'error', 'message': f'Caption reader: {error}'})

    def _run(self):
        from gpu_formatter import _attach_windows_kill_job, _close_windows_job
        from health import _read_memory
        try:
            if _read_memory()[0] < 2 * 2**30:
                raise MemoryError('Not enough RAM for provisional captions')
            self.process = subprocess.Popen([sys.executable, '-B', '-u',
                str(Path(__file__).with_name('streaming_worker.py')), json.dumps(self.terms)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.job = _attach_windows_kill_job(self.process)
            self.reader = threading.Thread(target=self._events, daemon=True, name='caption-reader')
            self.reader.start()
            def stderr():
                for line in self.process.stderr:
                    self.error_tail.append(line.decode('utf-8', 'replace')[-500:])
            threading.Thread(target=stderr, daemon=True, name='caption-errors').start()
            startup = time.monotonic()
            while not self.ready.wait(.2):
                if self.closed.is_set(): return
                if self.process.poll() is not None:
                    raise RuntimeError('Caption worker exited before becoming ready')
                if time.monotonic() - startup > 60:
                    raise TimeoutError('Caption model did not become ready within 60 seconds')
                if _read_memory()[0] < 2**30:
                    raise MemoryError('RAM headroom fell below 1 GiB during caption startup')
            while not self.closed.is_set():
                if self.process.poll() is not None:
                    raise RuntimeError('Caption worker exited')
                if _read_memory()[0] < 2**30:
                    raise MemoryError('RAM headroom fell below 1 GiB; captions stopped')
                try: packet = self.queue.get(timeout=.3)
                except queue.Empty: continue
                self.process.stdin.write(struct.pack('<I', 0xffffffff if packet is None else len(packet) // 4))
                if packet is not None: self.process.stdin.write(packet)
                self.process.stdin.flush()
            self.process.stdin.write(struct.pack('<I', 0))
            self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception as error:
            self.on_event({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
        finally:
            self.closed.set()
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            _close_windows_job(self.job)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=7)
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
