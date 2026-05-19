"""Mathematical-vector form of Vector-RLE.

Each token is a dimension; its accumulated frequency across the input is
the magnitude on that dimension. Order is discarded — this is the trade-off
versus SequenceRLE.

In exchange the structure supports real vector arithmetic: addition,
scalar multiplication, dot product, norm, and cosine similarity. Two RLE-
encoded streams can be compared directly without re-expanding them.
"""

from __future__ import annotations

import math
from typing import Dict, Generic, Hashable, Iterable, Mapping, TypeVar

from .bitpack import gamma_bits, leb128_bits
from .core import Run, encode_runs

T = TypeVar("T", bound=Hashable)


class MathRLE(Generic[T]):
    __slots__ = ("_mag",)

    def __init__(self, magnitudes: Mapping[T, int] | None = None) -> None:
        self._mag: Dict[T, int] = {}
        if magnitudes:
            for tok, m in magnitudes.items():
                if m < 0:
                    raise ValueError("magnitudes must be non-negative")
                if m:
                    self._mag[tok] = int(m)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_iterable(cls, seq: Iterable[T]) -> "MathRLE[T]":
        return cls.from_runs(encode_runs(seq))

    @classmethod
    def from_runs(cls, runs: Iterable[Run[T]]) -> "MathRLE[T]":
        mag: Dict[T, int] = {}
        for r in runs:
            mag[r.token] = mag.get(r.token, 0) + r.freq
        return cls(mag)

    @classmethod
    def from_sequence(cls, seq) -> "MathRLE[T]":
        """Collapse a SequenceRLE into a MathRLE (positional order is lost)."""
        return cls.from_runs(seq.runs)

    # -- access -------------------------------------------------------------

    @property
    def magnitudes(self) -> Dict[T, int]:
        return dict(self._mag)

    def tokens(self):
        return self._mag.keys()

    def __getitem__(self, token: T) -> int:
        return self._mag.get(token, 0)

    def __len__(self) -> int:
        return len(self._mag)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MathRLE):
            return NotImplemented
        return self._mag == other._mag

    def __repr__(self) -> str:
        return f"MathRLE({self._mag!r})"

    # -- arithmetic ---------------------------------------------------------

    def __add__(self, other: "MathRLE[T]") -> "MathRLE[T]":
        combined = dict(self._mag)
        for tok, m in other._mag.items():
            combined[tok] = combined.get(tok, 0) + m
        return MathRLE(combined)

    def __mul__(self, scalar: int) -> "MathRLE[T]":
        if scalar < 0:
            raise ValueError("scalar must be non-negative for unsigned magnitudes")
        if scalar == 0:
            return MathRLE()
        return MathRLE({tok: m * scalar for tok, m in self._mag.items()})

    __rmul__ = __mul__

    def dot(self, other: "MathRLE[T]") -> int:
        # iterate over the smaller mapping for efficiency
        a, b = (self._mag, other._mag) if len(self._mag) <= len(other._mag) else (other._mag, self._mag)
        total = 0
        for tok, m in a.items():
            other_m = b.get(tok)
            if other_m:
                total += m * other_m
        return total

    def norm(self) -> float:
        return math.sqrt(sum(m * m for m in self._mag.values()))

    def cosine_similarity(self, other: "MathRLE[T]") -> float:
        na = self.norm()
        nb = other.norm()
        if na == 0 or nb == 0:
            return 0.0
        return self.dot(other) / (na * nb)

    # -- bit accounting -----------------------------------------------------

    def bit_size(self, token_width: int, freq_encoding: str = "leb128") -> int:
        if freq_encoding not in ("leb128", "gamma", "raw32"):
            raise ValueError(f"unknown freq_encoding {freq_encoding!r}")
        total = 0
        for m in self._mag.values():
            total += token_width
            if freq_encoding == "leb128":
                total += leb128_bits(m)
            elif freq_encoding == "gamma":
                total += gamma_bits(m)
            else:  # raw32
                total += 32
        return total
