"""Bounded cache and VAD inference contracts, without model/network/GPU loads."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nano_experiment import BoundedGraphCache
from vad_benchmark import SileroProbabilities


def test_graph_cache_reuses_and_evicts_least_recent_shape():
    released = []
    cache = BoundedGraphCache(2, 100, release=lambda: released.append(1))
    a, b, c = object(), object(), object()
    assert cache.get("a", lambda: a, 40) is a
    assert cache.get("b", lambda: b, 40) is b
    assert cache.get("a", lambda: pytest.fail("cached graph rebuilt"), 40) is a
    assert cache.get("c", lambda: c, 40) is c
    assert list(cache.entries) == ["a", "c"]
    assert cache.evictions == 1 and cache.peak_entries == 2 and cache.peak_bytes == 80


def test_graph_cache_releases_memory_before_factory_and_limits_bytes():
    cache = BoundedGraphCache(10, 100)
    cache.get("a", object, 60)
    def factory():
        assert not cache.entries and cache.bytes == 0
        return object()
    cache.get("b", factory, 50)
    with pytest.raises(RuntimeError, match="single graph"):
        cache.get("too_large", lambda: pytest.fail("oversize graph built"), 101)
    assert list(cache.entries) == ["b"] and cache.bytes == 50


def test_evicted_graph_is_destroyed_before_another_is_allocated():
    import weakref
    class Graph:
        pass
    cache = BoundedGraphCache(1, 100)
    cache.get("a", Graph, 60)
    previous = weakref.ref(cache.entries["a"][0])
    def factory():
        assert previous() is None  # retaining the popped tuple would violate this
        return Graph()
    cache.get("b", factory, 60)


def test_vad_resets_state_per_piece_and_preserves_context():
    model = object.__new__(SileroProbabilities)
    calls = []
    class Session:
        def run(self, _, inputs):
            calls.append({k: v.copy() for k, v in inputs.items()})
            return np.array([[.9]], dtype=np.float32), inputs["state"] + 1
    model.session = Session()
    model(np.arange(700, dtype=np.float32))
    model(np.ones(100, dtype=np.float32))
    assert [c["input"].shape for c in calls] == [(1, 576)] * 3
    assert np.all(calls[0]["state"] == 0) and np.all(calls[1]["state"] == 1)
    assert np.all(calls[2]["state"] == 0)
    assert np.array_equal(calls[1]["input"][0, :64], np.arange(448, 512))
    assert np.all(calls[2]["input"][0, :64] == 0)


def test_cuda_gate_rejects_insufficient_headroom():
    from types import SimpleNamespace
    from recorder_probe import prepare_cuda
    cuda = SimpleNamespace(is_available=lambda: True, set_per_process_memory_fraction=lambda _: None,
                           mem_get_info=lambda: (2 * 2**30, 12 * 2**30))
    with pytest.raises(RuntimeError, match="free VRAM"):
        prepare_cuda(SimpleNamespace(cuda=cuda))


def test_formatter_scope_screen_catches_each_without_flagging_trial_duration():
    import json
    from formatter_benchmark import ROOT, screen
    cases = json.loads((ROOT / "tools/fixtures/formatter_cases.json").read_text())
    case = next(c for c in cases if c["id"] == "negation_contrast_quiz")
    assert not screen(case, "Group Cedar: 24 participants, completed 6 trials each.")["no_inferred_per_person_counts"]
    valid = "Group Maple completed 4 trials.\n- Each trial lasted 90 seconds."
    assert screen(case, valid)["no_inferred_per_person_counts"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree runner")
def test_probe_timeout_saves_partial_report_and_stops_child(tmp_path, monkeypatch):
    import json
    import psutil
    import recorder_probe
    import recorder_launcher
    monkeypatch.setattr(recorder_launcher, "recorder_state", lambda: {"running": False})
    script = tmp_path / "timed_child.py"
    script.write_text('import os,time\nprint(\'PROBE_RESULT {"pid":\'+str(os.getpid())+\'}\',flush=True)\ntime.sleep(30)\n')
    report = tmp_path / "timeout.json"
    with pytest.raises(RuntimeError, match="Hard timeout"):
        recorder_probe.run_probe(script, [], report, timeout=1.5, minimum_ram=.5)
    result = json.loads(report.read_text())
    assert result["stopped"].startswith("Hard timeout") and result["measurements"]
    child_pid = result["measurements"][0]["pid"]
    assert not psutil.pid_exists(child_pid)
    assert report.with_suffix(".progress.jsonl").read_text().strip()
