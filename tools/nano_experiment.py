"""Experiment-only Nano adapter: streamed weights, bounded graphs, existing guards.

The pinned upstream package belongs under ignored state, never on the recorder's
import path. CUDA graph caching is serial, matching the recorder inference lock.
"""
from collections import OrderedDict
import gc
from pathlib import Path
import sys


class BoundedGraphCache:
    def __init__(self, max_entries=2, max_bytes=384 * 2**20, release=lambda: None):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Graph cache limits must be positive")
        self.max_entries, self.max_bytes, self.release = max_entries, max_bytes, release
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.peak_entries = self.evictions = self.builds = 0

    def get(self, key, factory, estimated_bytes):
        if key in self.entries:
            value, size = self.entries.pop(key)
            self.entries[key] = value, size
            return value
        if estimated_bytes > self.max_bytes:
            raise RuntimeError("A single graph exceeds the configured persistent-buffer budget")
        evicted_any = False
        while self.entries and (len(self.entries) >= self.max_entries or self.bytes + estimated_bytes > self.max_bytes):
            _, (evicted, size) = self.entries.popitem(last=False)
            self.bytes -= size
            self.evictions += 1
            evicted_any = True
            del evicted
        if evicted_any:
            self.release()  # release evicted graph pools before building another shape
        value = factory()
        self.entries[key] = value, estimated_bytes
        self.bytes += estimated_bytes
        self.builds += 1
        self.peak_entries = max(self.peak_entries, len(self.entries))
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value


def install_streamed_loader():
    import nano_cohere_transcribe._loader as loader
    from safetensors import safe_open
    original = loader.safetensors_load_file
    def streamed(filename, device="cpu", **kwargs):
        if not str(device).startswith("cuda"):
            return original(filename, device=device, **kwargs)
        with safe_open(filename, framework="pt", device="cpu", backend="pread") as tensors:
            return {key: tensors.get_tensor(key).to(device) for key in tensors.keys()}
    loader.safetensors_load_file = streamed


def install_bounded_graphs(model, max_entries=2, max_bytes=384 * 2**20):
    import torch
    import nano_cohere_transcribe._graph as graph_module
    import nano_cohere_transcribe.generate as generate_module
    cache = BoundedGraphCache(max_entries, max_bytes, release=lambda: (gc.collect(), torch.cuda.empty_cache()))
    def get_graph(owner, B, T_enc, max_kv):
        if owner is not model:
            raise RuntimeError("Experiment graph cache belongs to another model")
        decoder = model.transf_decoder._decoder
        layers = len(decoder.layers)
        hidden = decoder.layers[0].first_sub_layer.hidden_size
        dtype = next(model.transf_decoder.parameters()).dtype
        item_bytes = torch.empty((), dtype=dtype).element_size()
        # Four KV banks plus masks, token/position buffers, and output logits.
        vocab = model.log_softmax.mlp.layer0.out_features
        estimated = 2 * layers * B * hidden * (max_kv + T_enc) * item_bytes
        estimated += (B * T_enc + B * vocab + 2) * item_bytes + (max_kv + 2 * B + 1) * 8
        key = B, T_enc, max_kv
        return cache.get(key, lambda: graph_module.DecoderStepGraph(
            model, B=B, T_enc=T_enc, max_kv=max_kv, dtype=dtype,
            device=next(model.parameters()).device), estimated)
    graph_module.get_or_build_graph = generate_module.get_or_build_graph = get_graph
    return cache


def load_backend(root, *, prompt="hf", max_graphs=2, graph_bytes=384 * 2**20, graphs=True):
    import numpy as np
    import torch
    import transcribe
    sys.path.insert(0, str(Path(root) / "state/efficiency_experiment/nano"))
    import nano_cohere_transcribe as nano
    from nano_cohere_transcribe.chunk import split_audio_chunks_energy
    install_streamed_loader()
    model = nano.from_pretrained(str(transcribe.COHERE_PATH), device="cuda", dtype=torch.bfloat16, warmup=False)
    model.config.max_audio_clip_s = transcribe.COHERE_CLIP_SECONDS
    cache = install_bounded_graphs(model, max_graphs, graph_bytes) if graphs else None
    if not graphs:
        import nano_cohere_transcribe.generate as generation
        original_generate = generation.greedy_generate
        def eager_generate(*args, **kwargs):
            kwargs["use_cuda_graph"] = False
            return original_generate(*args, **kwargs)
        generation.greedy_generate = eager_generate
    if prompt == "hf":
        original_prompt = model.tokenizer.build_prompt
        prefix = model.tokenizer.sp.piece_to_id("\u2581")
        if prefix == model.tokenizer.unk_id:
            raise RuntimeError("Cannot reproduce the recorder's leading prompt token")
        model.tokenizer.build_prompt = lambda *a, **kw: [prefix] + original_prompt(*a, **kw)
    class NanoBackend(transcribe._CohereBackend):
        name = "nano-cohere-experiment"
        def __init__(self):
            self.model, self._torch, self.graph_cache = model, torch, cache
            self.generated_tokens = 0
            self.device_desc = "cuda/bfloat16/" + ("graphs" if graphs else "eager") + "-experiment"
        def _pieces(self, audio):
            return split_audio_chunks_energy(np.asarray(audio, dtype=np.float32), 16000,
                                             transcribe.COHERE_CLIP_SECONDS, 5.0, 1600)
        def _piece_lengths(self, audio):
            return [len(p) for p in self._pieces(audio)]
        def _decode(self, audio, sample_rate, *, piece_samples=None):
            if sample_rate != 16000:
                raise ValueError("Expected 16 kHz audio")
            pieces = self._pieces(audio)
            rows = list(range(len(pieces)))
            if piece_samples is not None and piece_samples == [len(p) for p in pieces]:
                rows = [i for i, p in enumerate(pieces) if not transcribe.is_silent_audio(p, sample_rate)]
            texts = model._transcribe_chunks_batched([torch.from_numpy(pieces[i]) for i in rows],
                                                     language="en", punctuation=True, max_new_tokens=770,
                                                     batch_size=transcribe.COHERE_MAX_BATCH)
            if len(texts) != len(rows):
                raise RuntimeError("Nano returned an unexpected number of pieces")
            result = [""] * len(pieces)
            for i, text in zip(rows, texts):
                result[i] = text
            self.skipped_silent_pieces += len(pieces) - len(rows)
            return result
    return NanoBackend()
