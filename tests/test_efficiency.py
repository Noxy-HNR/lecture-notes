import json
import threading
import numpy as np
import main
import performance

def test_retry_restores_context_and_emits_once(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, 'PATH', tmp_path/'timings.jsonl')
    class Flaky:
        consecutive_silent_chunks = 0
        calls = 0
        def process(self, samples):
            assert self.consecutive_silent_chunks == 0
            self.calls += 1
            self.consecutive_silent_chunks = 5
            if self.calls == 1:
                raise RuntimeError('temporary')
            return []
    model = Flaky()
    got = []
    worker = main.TranscriptionWorker(model, lambda *args: got.append(args))
    worker.submit(np.zeros(16))
    worker.wait_idle()
    assert model.calls == 2 and len(got) == 1 and not worker.failures

def test_permanent_failure_has_recovery_range(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, 'PATH', tmp_path/'timings.jsonl')
    class Broken:
        def process(self, samples):
            raise RuntimeError('failure')
    path = tmp_path/'failed.json'
    worker = main.TranscriptionWorker(Broken(), lambda *args: None, failure_path=path)
    worker.submit(np.zeros(16))
    worker.wait_idle()
    assert json.loads(path.read_text())[0]['frames'] == 16
