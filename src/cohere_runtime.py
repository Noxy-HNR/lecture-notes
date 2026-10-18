"""Bounded loading and per-model CUDA graphs for the pinned Cohere runtime."""
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
import struct
import time


def checkpoint_layout(path):
    """Validate offsets and shapes before allocating GPU memory."""
    sizes = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8,
             'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    with Path(path).open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors header')
        length, = struct.unpack('<Q', prefix)
        if length > 64 * 2**20 or length < 2:
            raise ValueError('Invalid safetensors header size')
        header = handle.read(length)
        if len(header) != length:
            raise ValueError('Truncated safetensors metadata')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate checkpoint metadata key')
            result[key] = value
        return result
    entries = json.loads(header, object_pairs_hook=unique)
    entries.pop('__metadata__', None)
    if not entries:
        raise ValueError('Empty checkpoint')
    intervals = []
    for key, entry in entries.items():
        shape, dtype, offsets = entry['shape'], entry['dtype'], entry['data_offsets']
        if dtype not in sizes or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'Unsupported tensor layout: {key}')
        if len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
            raise ValueError(f'Invalid tensor offsets: {key}')
        start, end = offsets
        if end - start != math.prod(shape) * sizes[dtype] or start % sizes[dtype]:
            raise ValueError(f'Tensor size/alignment mismatch: {key}')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Checkpoint payload has gaps or overlaps')
        cursor = end
    data_start = 8 + length
    if Path(path).stat().st_size != data_start + cursor:
        raise ValueError('Checkpoint payload size mismatch')
    return entries, data_start, cursor


class WeightLoader:
    """128 MiB pinned pipeline; tensor-at-a-time fallback never stages the full model."""
    def __init__(self):
        self.stats = {}

    def __call__(self, filename, device='cpu', **kwargs):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        if torch.device(device).type != 'cuda':
            return load_file(filename, device=device, **kwargs)
        layout = checkpoint_layout(filename)
        free, _ = torch.cuda.mem_get_info(device)
        if free < layout[2] + 2**30:
            raise MemoryError('Insufficient free VRAM for the checkpoint plus 1 GiB headroom')
        reason = None
        try:
            return self._bulk(filename, device, layout)
        except (RuntimeError, MemoryError, OSError) as error:
            reason = f'{type(error).__name__}: {error}'
        # The failed frame/exception must be released before allocating a fallback.
        gc.collect()
        torch.cuda.empty_cache()
        with safe_open(filename, framework='pt', device='cpu', backend='pread') as tensors:
            state = {key: tensors.get_tensor(key).to(device) for key in tensors.keys()}
        self.stats = {'mode': 'streamed-fallback', 'reason': reason}
        return state

    def _bulk(self, filename, device, layout):
        import torch
        from health import _read_memory
        entries, data_start, size = layout
        chunk = 64 * 2**20
        available, _ = _read_memory()
        if available < 2 * chunk + 2**30:
            raise MemoryError('Insufficient available RAM for pinned staging')
        free, _ = torch.cuda.mem_get_info(device)
        if free < size + 2**30:
            raise MemoryError('Insufficient free VRAM for a contiguous checkpoint')
        began = time.perf_counter()
        blob = torch.empty(size, dtype=torch.uint8, device=device)
        hosts = [torch.empty(chunk, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        views = [memoryview(host.numpy()) for host in hosts]
        events = [None, None]
        stream = torch.cuda.Stream(device=device)
        try:
            with Path(filename).open('rb') as handle:
                handle.seek(data_start)
                cursor, slot = 0, 0
                while cursor < size:
                    if events[slot] is not None:
                        events[slot].synchronize()
                    take = min(chunk, size - cursor)
                    if handle.readinto(views[slot][:take]) != take:
                        raise OSError('Short checkpoint read')
                    with torch.cuda.stream(stream):
                        blob[cursor:cursor + take].copy_(hosts[slot][:take], non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(stream)
                        events[slot] = event
                    cursor += take
                    slot = 1 - slot
        finally:
            # Also protects host buffers if reading fails while a transfer is in flight.
            stream.synchronize()
        types = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
                 'F64': torch.float64, 'I64': torch.int64, 'I32': torch.int32,
                 'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool}
        result = {}
        for key, entry in entries.items():
            start, end = entry['data_offsets']
            result[key] = blob[start:end].view(types[entry['dtype']]).reshape(entry['shape'])
        self.stats = {'mode': 'bulk', 'payload_bytes': size, 'staging_bytes': 2 * chunk,
                      'seconds': time.perf_counter() - began}
        return result


class GraphCache:
    def __init__(self, max_entries=2, max_bytes=384 * 2**20):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries = OrderedDict()
        self.bytes = self.builds = self.evictions = 0
        self.peak_bytes = self.peak_entries = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def get(self, key, factory, estimated):
        if key in self.entries:
            value = self.entries.pop(key)
            self.entries[key] = value
            return value[0]
        if estimated > self.max_bytes:
            raise RuntimeError('Decoder graph exceeds the persistent-buffer budget')
        evicted_any = False
        while self.entries and (len(self.entries) >= self.max_entries or self.bytes + estimated > self.max_bytes):
            _, (evicted, size) = self.entries.popitem(last=False)
            self.bytes -= size
            self.evictions += 1
            del evicted
            evicted_any = True
        if evicted_any:
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        value = factory()
        self.entries[key] = value, estimated
        self.bytes += estimated
        self.builds += 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.peak_entries = max(self.peak_entries, len(self.entries))
        return value


def install_graphs(model):
    import torch
    from vendor.nano_cohere_transcribe._graph import DecoderStepGraph
    cache = GraphCache()
    def get_graph(owner, B, T_enc, max_kv):
        if owner is not model:
            raise RuntimeError('Decoder graph belongs to another model')
        layers = model.transf_decoder._decoder.layers
        hidden = layers[0].first_sub_layer.hidden_size
        dtype = next(model.transf_decoder.parameters()).dtype
        item = torch.empty((), dtype=dtype).element_size()
        vocab = model.log_softmax.mlp.layer0.out_features
        estimated = 2 * len(layers) * B * hidden * (max_kv + T_enc) * item
        estimated += (B * T_enc + B * vocab + 2) * item + (max_kv + 2 * B + 1) * 8
        return cache.get((B, T_enc, max_kv), lambda: DecoderStepGraph(
            model, B=B, T_enc=T_enc, max_kv=max_kv, dtype=dtype,
            device=next(model.parameters()).device), estimated)
    model.decoder_graph_factory = get_graph
    return cache
