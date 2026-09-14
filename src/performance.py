"""Low-overhead stage timings; no transcript text is logged."""
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'state' / 'performance.jsonl'
_lock = threading.Lock()

@contextmanager
def stage(name):
    started = time.perf_counter()
    ok = False
    try:
        yield
        ok = True
    finally:
        try:
            with _lock:
                PATH.parent.mkdir(parents=True, exist_ok=True)
                with PATH.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'stage': name, 'seconds': time.perf_counter()-started,
                                             'ok': ok, 'at': time.time()})+'\n')
        except OSError:
            pass
