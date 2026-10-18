"""Public ``from_pretrained`` entry point and small dataclass wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ._loader import load_model_from_snapshot, resolve_snapshot
from .audio import load_audio_16k_mono
from .model import CohereAsr

_MODEL_CACHE: dict[tuple[str, str, str], CohereAsr] = {}


@dataclass
class TranscriptionResult:
    text: str
    token_ids: list[int]
    language: str


def from_pretrained(
    repo_id_or_path: str = "CohereLabs/cohere-transcribe-03-2026",
    device: str | torch.device = "cuda",
    dtype: Optional[torch.dtype] = None,
    warmup: bool = True,
    decoder_tokenizer: str = "sentencepiece",
) -> CohereAsr:
    """Download or load a Cohere ASR checkpoint and return a ready-to-use :class:`CohereAsr`.

    ``decoder_tokenizer`` picks the detokenizer backend: ``"sentencepiece"``
    (default, bundled C++ SP) or ``"fast"`` (HF Rust ``tokenizers`` via
    ``AutoTokenizer(use_fast=True)``). Both are fast; use ``"fast"`` for
    consistency with HF-based pipelines.

    Subsequent calls with the same ``(repo_id, device, dtype, decoder_tokenizer)``
    return the cached instance.
    """
    key = (str(repo_id_or_path), str(device), str(dtype), decoder_tokenizer)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    snap = resolve_snapshot(repo_id_or_path)
    model, tokenizer = load_model_from_snapshot(
        snap, device=device, dtype=dtype, decoder_tokenizer=decoder_tokenizer
    )
    model.tokenizer = tokenizer
    model.snapshot_dir = Path(snap)

    if warmup:
        _warmup(model)

    _MODEL_CACHE[key] = model
    return model


def _warmup(model: CohereAsr) -> None:
    device = next(model.parameters()).device
    # 1 s of silence is enough to exercise every kernel path.
    dummy = torch.zeros(16000, dtype=torch.float32, device=device)
    _ = model.transcribe(dummy, language="en", max_new_tokens=1)


def transcribe_file(
    model: CohereAsr,
    audio_path: str,
    language: str = "en",
    punctuation: bool = True,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> TranscriptionResult:
    waveform = load_audio_16k_mono(audio_path)
    text = model.transcribe(
        waveform,
        language=language,
        punctuation=punctuation,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    return TranscriptionResult(text=text, token_ids=[], language=language)
