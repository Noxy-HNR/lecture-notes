"""SentencePiece / HF-fast tokenizer wrapper for Cohere ASR.

All control tokens (``<|startoftranscript|>``, ``<|en|>``, ``<|pnc|>``, etc.)
live in the SentencePiece vocabulary, so encoding the prompt is a simple
``piece_to_id`` lookup — we never have to go through the HF tokenizer's fast
encoder (which character-BPEs control tokens on ``transformers >= 5.4``).

Decoding a stream of token ids can use either:
- ``sentencepiece.SentencePieceProcessor.DecodeIds`` (C++, default), or
- a HuggingFace ``tokenizers``-backed fast tokenizer via
  ``AutoTokenizer(..., use_fast=True).decode`` (pass ``decoder="fast"`` to
  :class:`CohereTokenizer`).

Both are fast — SP and the Rust fast tokenizer both decode at ~µs/token. Swap
between them for consistency with existing HF stacks or to validate parity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import sentencepiece as spm

BOS = "<|startoftranscript|>"
EOS = "<|endoftext|>"
PAD = "<pad>"
UNK = "<unk>"
STARTOFCONTEXT = "<|startofcontext|>"
EMO_UNDEFINED = "<|emo:undefined|>"
NOTIMESTAMP = "<|notimestamp|>"
NODIARIZE = "<|nodiarize|>"
NOITN = "<|noitn|>"
PNC = "<|pnc|>"
NOPNC = "<|nopnc|>"

SUPPORTED_LANGUAGES = (
    "en",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "el",
    "ar",
    "ja",
    "zh",
    "vi",
    "ko",
)


class CohereTokenizer:
    def __init__(
        self,
        spm_model_file: str,
        decoder: Literal["sentencepiece", "fast"] = "sentencepiece",
        snapshot_dir: str | Path | None = None,
    ):
        """Args:
            spm_model_file: path to tokenizer.model (SentencePiece).
            decoder: which backend to use for :meth:`decode`. ``"sentencepiece"``
                (default) uses the bundled C++ SP processor; ``"fast"`` uses
                HuggingFace's Rust fast tokenizer loaded from ``snapshot_dir``.
            snapshot_dir: directory containing ``tokenizer.json`` and friends;
                required when ``decoder="fast"``. If ``None``, defaults to the
                directory of ``spm_model_file``.
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model_file)
        self.bos_id = self.sp.piece_to_id(BOS)
        self.eos_id = self.sp.piece_to_id(EOS)
        self.pad_id = self.sp.piece_to_id(PAD)
        self.unk_id = self.sp.piece_to_id(UNK)
        # Cache ids of all control tokens so decode(skip_special_tokens=True)
        # can filter them cheaply.
        self._special_ids = self._build_special_id_set()

        self._fast = None
        if decoder == "fast":
            from transformers import AutoTokenizer
            d = Path(snapshot_dir) if snapshot_dir else Path(spm_model_file).parent
            self._fast = AutoTokenizer.from_pretrained(
                d.as_posix(), trust_remote_code=True, use_fast=True
            )
        elif decoder != "sentencepiece":
            raise ValueError(f"decoder must be 'sentencepiece' or 'fast', got {decoder!r}")
        self._decoder = decoder

    def _build_special_id_set(self) -> set[int]:
        ids: set[int] = set()
        n = self.sp.get_piece_size()
        for i in range(n):
            piece = self.sp.id_to_piece(i)
            if piece.startswith("<") and piece.endswith(">"):
                ids.add(i)
        return ids

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def build_prompt(self, language: str, punctuation: bool = True) -> list[int]:
        """Return the decoder prompt prefix as token ids.

        Matches ``CohereAsrForConditionalGeneration.build_prompt``:
        ``<|startofcontext|><|startoftranscript|><|emo:undefined|><|lang|><|lang|>
        <|pnc|or|nopnc|><|noitn|><|notimestamp|><|nodiarize|>``
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}. Supported: {sorted(SUPPORTED_LANGUAGES)}"
            )
        lang_tok = f"<|{language}|>"
        pieces = [
            STARTOFCONTEXT,
            BOS,
            EMO_UNDEFINED,
            lang_tok,
            lang_tok,
            PNC if punctuation else NOPNC,
            NOITN,
            NOTIMESTAMP,
            NODIARIZE,
        ]
        return [self.sp.piece_to_id(p) for p in pieces]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        ids = list(token_ids)
        # Chop off trailing EOS-and-later, then strip specials (cheap int filter),
        # then hand off to whichever backend we were configured with.
        if skip_special_tokens:
            try:
                ids = ids[: ids.index(self.eos_id)]
            except ValueError:
                pass
            ids = [i for i in ids if i not in self._special_ids]
        if self._fast is not None:
            # HF fast tokenizer (tokenizers lib). skip_special_tokens is handled
            # by us above, so pass False to avoid double-filtering.
            return self._fast.decode(ids, skip_special_tokens=False)
        return self.sp.DecodeIds(ids)
