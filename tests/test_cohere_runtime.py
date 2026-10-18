import json
import struct
import weakref
from types import SimpleNamespace

import pytest

from cohere_runtime import checkpoint_layout, GraphCache, WeightLoader
import transcribe


def checkpoint(tmp_path, entries, payload=b'\0' * 8):
    header = json.dumps(entries).encode()
    path = tmp_path / 'weights.safetensors'
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
    return path


def test_layout_validates_before_allocation(tmp_path):
    path = checkpoint(tmp_path, {'x': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}})
    entries, start, size = checkpoint_layout(path)
    assert size == 8 and start + size == path.stat().st_size
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='size mismatch'):
        checkpoint_layout(path)


@pytest.mark.parametrize('shape,offsets', [([3], [0, 8]), ([2], [4, 12]), ([-2], [0, 8])])
def test_malformed_layout_rejected(tmp_path, shape, offsets):
    path = checkpoint(tmp_path, {'x': {'dtype': 'F32', 'shape': shape, 'data_offsets': offsets}})
    with pytest.raises(ValueError):
        checkpoint_layout(path)


def test_graph_evicted_before_replacement(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    cache = GraphCache(max_entries=1, max_bytes=10)
    class Graph: pass
    old = weakref.ref(cache.get('a', Graph, 8))
    def replacement():
        assert old() is None
        return Graph()
    cache.get('b', replacement, 8)
    assert cache.bytes == 8 and cache.evictions == 1


def test_runtime_fallback_and_explicit_selection(monkeypatch):
    reference = SimpleNamespace(device_desc='reference')
    monkeypatch.setattr(transcribe, '_CohereBackend', lambda: reference)
    def unavailable(): raise RuntimeError('graph runtime unavailable')
    monkeypatch.setattr(transcribe, '_NanoBackend', unavailable)
    monkeypatch.setenv('NOTE_RECORDER_ASR', 'auto')
    assert transcribe._load_backend() is reference
    assert 'runtime unavailable' in reference.device_desc
    monkeypatch.setenv('NOTE_RECORDER_ASR', 'nano')
    with pytest.raises(RuntimeError, match='runtime unavailable'):
        transcribe._load_backend()
    monkeypatch.setenv('NOTE_RECORDER_ASR', 'hf')
    assert transcribe._load_backend() is reference


def test_failed_bulk_frame_is_freed_before_streamed_fallback(tmp_path, monkeypatch):
    import sys
    from contextlib import contextmanager
    path = checkpoint(tmp_path, {'x': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}})
    references = []
    class Allocation: pass
    loader = WeightLoader()
    def failed(*args):
        allocation = Allocation()
        references.append(weakref.ref(allocation))
        raise RuntimeError('contiguous allocation failed')
    monkeypatch.setattr(loader, '_bulk', failed)
    torch = SimpleNamespace(device=lambda d: SimpleNamespace(type='cuda'),
                            cuda=SimpleNamespace(empty_cache=lambda: None,
                                                 mem_get_info=lambda d: (10 * 2**30, 12 * 2**30)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    @contextmanager
    def opened(*args, **kwargs):
        assert references[0]() is None
        assert kwargs['backend'] == 'pread'
        yield SimpleNamespace(keys=lambda: ['x'], get_tensor=lambda key:
            SimpleNamespace(to=lambda device: 'gpu tensor'))
    monkeypatch.setitem(sys.modules, 'safetensors', SimpleNamespace(safe_open=opened))
    monkeypatch.setitem(sys.modules, 'safetensors.torch', SimpleNamespace(load_file=lambda *a, **k: None))
    assert loader(path, 'cuda') == {'x': 'gpu tensor'}
    assert loader.stats['mode'] == 'streamed-fallback'


def test_insufficient_vram_rejects_loading_before_allocation(tmp_path, monkeypatch):
    import sys
    path = checkpoint(tmp_path, {'x': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        device=lambda d: SimpleNamespace(type='cuda'),
        cuda=SimpleNamespace(mem_get_info=lambda d: (2**30, 12 * 2**30))))
    monkeypatch.setitem(sys.modules, 'safetensors.torch', SimpleNamespace(load_file=None))
    loader = WeightLoader()
    def forbidden(*args): pytest.fail('allocated before VRAM admission')
    monkeypatch.setattr(loader, '_bulk', forbidden)
    with pytest.raises(MemoryError, match='1 GiB headroom'):
        loader(path, 'cuda')
