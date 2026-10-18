"""Energy-based audio chunking for long-form transcription.

Long audio (> ``max_audio_clip_s``, default 35 s) doesn't fit the model's
decoder-side positional encoding budget and cross-attention mask — it has
to be split. This ports ``split_audio_chunks_energy`` from the reference
``modeling_cohere_asr.py``: each chunk is at most ``max_audio_clip_s`` long,
and boundaries are chosen at the quietest point within the last
``overlap_chunk_second`` of each window so we don't cut in the middle of a
word.

No reimplementation liberties taken — layout and numerics match upstream.
"""
from __future__ import annotations

import numpy as np

NO_SPACE_LANGS = frozenset({"ja", "zh"})


def split_audio_chunks_energy(
    waveform: np.ndarray,
    sample_rate: int,
    max_audio_clip_s: float,
    overlap_chunk_second: float,
    min_energy_window_samples: int,
) -> list[np.ndarray]:
    """Split a 1-D waveform into <=``max_audio_clip_s`` chunks at quiet points."""
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform (1D), got shape={waveform.shape}")
    chunk_size = max(1, int(round(max_audio_clip_s * sample_rate)))
    # In energy-split mode, `overlap_chunk_second` is the search window near the
    # boundary (NeMo parity), not literal waveform overlap between chunks.
    boundary_context_size = max(1, int(round(overlap_chunk_second * sample_rate)))
    total = waveform.shape[0]
    if total <= chunk_size:
        return [waveform.copy()]

    chunks: list[tuple[int, int]] = []
    idx = 0
    while idx < total:
        if idx + chunk_size >= total:
            chunks.append((idx, total))
            break
        search_start = max(idx, idx + chunk_size - boundary_context_size)
        search_end = min(idx + chunk_size, total)
        if search_end <= search_start:
            split_point = idx + chunk_size
        else:
            split_point = _find_split_point_energy(
                waveform,
                start_idx=search_start,
                end_idx=search_end,
                min_energy_window_samples=min_energy_window_samples,
            )
        split_point = max(idx + 1, min(split_point, total))
        chunks.append((idx, split_point))
        idx = split_point
    return [waveform[s:e].copy() for s, e in chunks if e > s]


def _find_split_point_energy(
    waveform: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_energy_window_samples: int,
) -> int:
    """Return the sample index within [start_idx, end_idx) with lowest RMS energy."""
    segment = waveform[start_idx:end_idx]
    if segment.shape[0] <= min_energy_window_samples:
        return (start_idx + end_idx) // 2
    min_energy = float("inf")
    quietest = start_idx
    upper = segment.shape[0] - min_energy_window_samples
    for i in range(0, upper, min_energy_window_samples):
        win = segment[i : i + min_energy_window_samples]
        e = float(np.sqrt(np.mean(win * win)))
        if e < min_energy:
            min_energy = e
            quietest = start_idx + i
    return quietest


def join_chunk_texts(texts: list[str], separator: str = " ") -> str:
    parts = [p.strip() for p in texts if p and p.strip()]
    return separator.join(parts) if parts else ""


def get_chunk_separator(language: str) -> str:
    return "" if language in NO_SPACE_LANGS else " "
