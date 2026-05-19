"""Vocabulary: the geometric dictionary.

A `Vocabulary` maps tokens (`bytes`) to IDs (`int`) and back. Tokens
are stored in descending frequency order, so the most common token
gets ID 0, the second most common ID 1, and so on. This is the optimal
arrangement for entropy-coded positional indices (small IDs are
cheaper because the matching frequency is large).

The `freqs` property exposes the **magnitude vector** — the count of
each token in the source. This is the same vector that the entropy
coder needs as its probability model when coding the positional index,
which is the v5 unification: one geometric object, two roles.

Serialization layout:

    leb128(n_tokens)
    leb128(total_token_count)        # sum of frequencies; also the
                                     # length of the positional index
    for each unique token (in ID order):
        leb128(len(token))
        bytes(token)
        leb128(freq)                 # the magnitude

Frequencies are written into the vocabulary because the decoder needs
them for rANS. They are part of the geometric dictionary by design.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .bitpack import leb128_decode, leb128_encode


class Vocabulary:
    __slots__ = ("_tokens", "_freqs", "_index", "_total")

    def __init__(self, tokens: List[bytes], freqs: List[int]) -> None:
        if len(tokens) != len(freqs):
            raise ValueError("tokens and freqs must be the same length")
        self._tokens = list(tokens)
        self._freqs = list(freqs)
        self._index: Dict[bytes, int] = {t: i for i, t in enumerate(self._tokens)}
        self._total = sum(self._freqs)

    @classmethod
    def from_tokens(cls, tokens: List[bytes]) -> "Vocabulary":
        counts: Dict[bytes, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        # Sort by descending frequency, breaking ties by token bytes for
        # deterministic IDs across encoder/decoder.
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return cls([k for k, _ in ordered], [v for _, v in ordered])

    # -- access -----------------------------------------------------------

    @property
    def tokens(self) -> List[bytes]:
        return list(self._tokens)

    @property
    def freqs(self) -> List[int]:
        return list(self._freqs)

    @property
    def total(self) -> int:
        return self._total

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> bytes:
        return self._tokens[idx]

    def tokens_to_ids(self, tokens: List[bytes]) -> List[int]:
        idx = self._index
        return [idx[t] for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[bytes]:
        tk = self._tokens
        return [tk[i] for i in ids]

    # -- serialization ---------------------------------------------------

    def serialize(self) -> bytes:
        out = bytearray()
        out += leb128_encode(len(self._tokens))
        out += leb128_encode(self._total)
        for tok, freq in zip(self._tokens, self._freqs):
            out += leb128_encode(len(tok))
            out += tok
            out += leb128_encode(freq)
        return bytes(out)

    @classmethod
    def deserialize(cls, buf: bytes, pos: int = 0) -> Tuple["Vocabulary", int]:
        n_tokens, pos = leb128_decode(buf, pos)
        total, pos = leb128_decode(buf, pos)
        tokens: List[bytes] = []
        freqs: List[int] = []
        for _ in range(n_tokens):
            length, pos = leb128_decode(buf, pos)
            tokens.append(bytes(buf[pos : pos + length]))
            pos += length
            freq, pos = leb128_decode(buf, pos)
            freqs.append(freq)
        vocab = cls(tokens, freqs)
        if vocab.total != total:
            raise ValueError(
                f"vocabulary total mismatch: stored {total}, computed {vocab.total}"
            )
        return vocab, pos
