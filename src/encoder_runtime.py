"""Optional encoder compilation; the durable recorder can always recover eagerly."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(model, mode=None):
    mode = mode or os.environ.get('NOTE_RECORDER_ENCODER', 'eager')
    if mode == 'eager':
        return {'mode': 'eager'}
    if mode not in ('compile-bf16', 'compile-fp8'):
        raise ValueError(f'Unknown encoder mode: {mode}')
    import torch
    import triton  # fail before changing the encoder if the Windows compiler is absent
    from health import _read_memory
    if _read_memory()[0] < 3 * 2**30:
        raise MemoryError('Encoder compilation needs 3 GiB available RAM')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / 'state' / 'compiled_encoder'))
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
    os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / 'state' / 'compiled_encoder' / 'triton'))
    encoder = model.encoder
    stats = {'mode': mode, 'compiled_layers': 0, 'fallbacks': 0,
             'batch_policy': 'single-piece' if mode == 'compile-bf16' else 'experimental-all-batches'}
    originals = [(layer, layer.forward) for layer in encoder.layers]
    linear_originals = []
    if mode == 'compile-fp8':
        needed = sum(linear.weight.numel() for layer in encoder.layers for ff in
                     (layer.feed_forward1, layer.feed_forward2) for linear in (ff.linear1, ff.linear2))
        if torch.cuda.mem_get_info()[0] < needed + 2**30:
            raise MemoryError('Insufficient VRAM for FP8 candidate weights plus 1 GiB headroom')
        # Keep the reference BF16 weights for exact eager recovery. This candidate
        # uses native scaled FP8 matrix multiplication only for the large FFNs.
        for layer in encoder.layers:
            for ff in (layer.feed_forward1, layer.feed_forward2):
                for linear in (ff.linear1, ff.linear2):
                    linear_originals.append((linear, linear.forward))
                    _install_fp8(linear, torch)
    for layer in encoder.layers:
        eager = layer.forward
        compiled = torch.compile(eager, dynamic=True, fullgraph=True, mode='default')
        def guarded(*args, _compiled=compiled, _eager=eager, **kwargs):
            if mode == 'compile-bf16' and args[0].shape[0] != 1:
                return _eager(*args, **kwargs)
            try:
                return _compiled(*args, **kwargs)
            except Exception as error:
                stats['fallbacks'] += 1
                stats['reason'] = f'{type(error).__name__}: {error}'[:1000]
                # Restore every layer together; repeated compiler attempts add latency.
                for saved_layer, saved_forward in originals:
                    saved_layer.forward = saved_forward
                for linear, saved_forward in linear_originals:
                    linear.forward = saved_forward
                return _eager(*args, **kwargs)
        layer.forward = guarded
        stats['compiled_layers'] += 1
    return stats


def _install_fp8(linear, torch):
    weight = linear.weight
    scale = weight.detach().abs().amax().clamp(min=1e-8) / 448
    quantized = (weight.detach() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    original = linear.forward
    def forward(x):
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[1] % 16 or weight.shape[0] % 16:
            return original(x)
        rows = flat.shape[0]
        flat = torch.nn.functional.pad(flat, (0, 0, 0, (-rows) % 16))
        activation_scale = flat.abs().amax().float().clamp(min=1e-8) / 448
        activation = (flat / activation_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        result = torch._scaled_mm(activation, quantized.t(), scale_a=activation_scale,
                                  scale_b=scale.float(), bias=linear.bias, out_dtype=x.dtype)
        return result[:rows].reshape(*x.shape[:-1], weight.shape[0])
    linear.forward = forward
