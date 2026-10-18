"""Log-mel filterbank features matching NeMo's FilterbankFeatures.

Mirrors `processing_cohere_asr.FilterbankFeatures` with only the paths the
Cohere-transcribe checkpoint actually uses (preemph=0.97, mag_power=2, log-add
guard, per-feature normalize, pad_to=16). The mel filterbank and STFT window
are loaded from the safetensors checkpoint as buffers (`preprocessor.featurizer.fb`
and `preprocessor.featurizer.window`), so librosa is NOT required at runtime.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DITHER_CONSTANT = 1e-5
LOG_ZERO_GUARD = 2**-24


class FilterbankFeatures(nn.Module):
    """Minimal log-mel feature extractor.

    Input: ``x`` of shape ``[B, T_samples]`` and ``seq_len`` of shape ``[B]``.
    Output: ``features`` of shape ``[B, n_mels, T_frames]`` and new ``lengths``.
    """

    window: torch.Tensor
    fb: torch.Tensor

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 400,
        n_window_stride: int = 160,
        n_fft: int = 512,
        nfilt: int = 128,
        preemph: float = 0.97,
        dither: float = DITHER_CONSTANT,
        log_zero_guard_value: float = LOG_ZERO_GUARD,
        mag_power: float = 2.0,
        pad_to: int = 16,
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft
        self.nfilt = nfilt
        self.preemph = preemph
        self.dither = dither
        self.log_zero_guard_value = log_zero_guard_value
        self.mag_power = mag_power
        self.pad_to = pad_to
        self.pad_value = pad_value

        # Placeholder buffers; real values come from the checkpoint via
        # `preprocessor.featurizer.{window,fb}`.
        self.register_buffer("window", torch.hann_window(n_window_size, periodic=False), persistent=False)
        self.register_buffer("fb", torch.zeros(1, nfilt, n_fft // 2 + 1), persistent=False)

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(0)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        pad_amount = (self.n_fft // 2) * 2
        seq_len = torch.floor_divide(seq_len + pad_amount - self.n_fft, self.hop_length)
        return seq_len.to(dtype=torch.long)

    def _apply_dither(self, x: torch.Tensor, seq_len_time: torch.Tensor) -> torch.Tensor:
        if self.dither <= 0:
            return x
        # Deterministic per-sample dither, seeded by valid length (matches reference).
        gen = torch.Generator(device=x.device if x.device.type == "cpu" else "cpu")
        for i in range(x.shape[0]):
            valid = min(int(seq_len_time[i].item()), x.shape[1])
            if valid <= 0:
                continue
            gen.manual_seed(valid)
            noise = torch.randn((valid,), dtype=x.dtype, device="cpu", generator=gen)
            x[i, :valid] = x[i, :valid] + self.dither * noise.to(x.device)
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
        seq_len_time = seq_len
        seq_len = self.get_seq_len(seq_len)

        x = self._apply_dither(x, seq_len_time)

        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        with torch.amp.autocast(x.device.type, enabled=False):
            stft = torch.stft(
                x.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=True,
                window=self.window.to(dtype=torch.float, device=x.device),
                return_complex=True,
                pad_mode="constant",
            )
        mag = torch.view_as_real(stft)
        mag = torch.sqrt(mag.pow(2).sum(-1))
        if self.mag_power != 1.0:
            mag = mag.pow(self.mag_power)

        with torch.amp.autocast(x.device.type, enabled=False):
            fb = self.fb.to(dtype=mag.dtype, device=mag.device)
            feats = torch.matmul(fb, mag)

        feats = torch.log(feats + self.log_zero_guard_value)

        # Per-feature normalization (masked mean/std over time).
        feats = _per_feature_normalize(feats, seq_len)

        # Mask padded frames.
        max_len = feats.size(-1)
        mask = torch.arange(max_len, device=feats.device).unsqueeze(0) >= seq_len.unsqueeze(1)
        feats = feats.masked_fill(mask.unsqueeze(1), self.pad_value)

        if self.pad_to > 0:
            rem = feats.size(-1) % self.pad_to
            if rem != 0:
                feats = F.pad(feats, (0, self.pad_to - rem), value=self.pad_value)
        return feats, seq_len


def _per_feature_normalize(x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    batch, n_mels, max_time = x.shape
    time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch, max_time)
    valid = time_steps < seq_len.unsqueeze(1)  # [B, T]
    valid_f = valid.unsqueeze(1)  # [B, 1, T]
    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(1)  # [B, 1]
    mean = torch.where(valid_f, x, torch.zeros_like(x)).sum(dim=2) / denom  # [B, n_mels]
    diff = x - mean.unsqueeze(2)
    var = torch.where(valid_f, diff * diff, torch.zeros_like(diff)).sum(dim=2) / (denom - 1).clamp(min=1)
    std = torch.sqrt(var)
    std = std.masked_fill(std.isnan(), 0.0) + DITHER_CONSTANT
    return diff / std.unsqueeze(2)
