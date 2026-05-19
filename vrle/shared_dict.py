"""Shared base dictionary for ``vector_rle_shared``.

A ``SharedDict`` is an ordered list of byte tokens. Each token's
position in the list is its ID; the first token has ID 0. The
``vector_rle_shared`` pipeline references tokens in this dict by ID
without re-shipping their bytes.

Serialization is intentionally trivial: ``leb128(version) ||
leb128(n_tokens) || for each token: leb128(len) || bytes(token)``.
The format mirrors the inline vocabulary used by ``vector_rle`` so a
shared dict file is identical in structure to what a self-contained
file would have stored inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .bitpack import leb128_decode, leb128_encode


class SharedDict:
    __slots__ = ("tokens", "_forward", "version")

    def __init__(self, tokens: List[bytes], version: int = 1) -> None:
        self.tokens = list(tokens)
        self._forward: Dict[bytes, int] = {t: i for i, t in enumerate(self.tokens)}
        if len(self._forward) != len(self.tokens):
            raise ValueError("SharedDict tokens must be unique")
        self.version = version

    def __len__(self) -> int:
        return len(self.tokens)

    def get_id(self, token: bytes) -> Optional[int]:
        return self._forward.get(token)

    def get_token(self, idx: int) -> bytes:
        return self.tokens[idx]

    def __contains__(self, token: bytes) -> bool:
        return token in self._forward

    # ------------------------------------------------------------------

    def serialize(self) -> bytes:
        out = bytearray()
        out += leb128_encode(self.version)
        out += leb128_encode(len(self.tokens))
        for t in self.tokens:
            out += leb128_encode(len(t))
            out += t
        return bytes(out)

    @classmethod
    def deserialize(cls, buf: bytes, pos: int = 0) -> "SharedDict":
        version, pos = leb128_decode(buf, pos)
        n, pos = leb128_decode(buf, pos)
        tokens: List[bytes] = []
        for _ in range(n):
            L, pos = leb128_decode(buf, pos)
            tokens.append(bytes(buf[pos : pos + L]))
            pos += L
        return cls(tokens, version)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.serialize())

    @classmethod
    def load(cls, path: str | Path) -> "SharedDict":
        return cls.deserialize(Path(path).read_bytes())
