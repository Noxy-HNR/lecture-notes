import sys
from types import SimpleNamespace

import encoder_runtime
import health


def setup_compiler(monkeypatch, compile_fn):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(compile=compile_fn))
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace())
    monkeypatch.setattr(health, '_read_memory', lambda: (4 * 2**30, 16 * 2**30))
    # Do not let the test set compiler-cache configuration for subsequent tests.
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', 'unused-test-cache')
    monkeypatch.setenv('TORCHINDUCTOR_COMPILE_THREADS', '1')
    monkeypatch.setenv('TRITON_CACHE_DIR', 'unused-test-cache')


def test_compiler_failure_restores_all_layers_and_stops_retries(monkeypatch):
    calls = []
    def eager(value):
        return value
    layers = [SimpleNamespace(forward=eager), SimpleNamespace(forward=eager)]
    def compile_fn(function, **kwargs):
        def failed(*args, **kwargs):
            calls.append('compiled')
            raise RuntimeError('compiler kernel failed')
        return failed
    setup_compiler(monkeypatch, compile_fn)
    model = SimpleNamespace(encoder=SimpleNamespace(layers=layers))
    stats = encoder_runtime.configure(model, 'compile-bf16')
    value = SimpleNamespace(shape=(1, 10, 128))
    assert layers[0].forward(value) is value
    assert stats['fallbacks'] == 1 and 'compiler kernel failed' in stats['reason']
    assert all(layer.forward is eager for layer in layers)
    assert layers[1].forward(value) is value
    assert calls == ['compiled']


def test_larger_batches_use_eager_without_triggering_compilation(monkeypatch):
    def compile_fn(function, **kwargs):
        def forbidden(*args, **kwargs):
            raise AssertionError('large batch entered the compiler')
        return forbidden
    setup_compiler(monkeypatch, compile_fn)
    layer = SimpleNamespace(forward=lambda value: value)
    model = SimpleNamespace(encoder=SimpleNamespace(layers=[layer]))
    stats = encoder_runtime.configure(model, 'compile-bf16')
    value = SimpleNamespace(shape=(4, 10, 128))
    assert layer.forward(value) is value
    assert stats['fallbacks'] == 0
