"""Ordered Vector-RLE: preserves the input sequence's order.

This is the form that round-trips losslessly. Compression comes from
collapsing adjacent equal tokens into (token, freq) runs and storing the
frequency with a variable-length code.
"""

from __future__ import annotations

from typing import Generic, Hashable, Iterable, Iterator, List, TypeVar

from .bitpack import gamma_bits, leb128_bits
from .core import Run, decode_runs, encode_runs, iter_runs

T = TypeVar("T", bound=Hashable)


class SequenceRLE(Generic[T]):
    __slots__ = ("_runs",)

    def __init__(self, runs: List[Run[T]] | None = None) -> None:
        self._runs = list(runs) if runs else []

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_iterable(cls, seq: Iterable[T]) -> "SequenceRLE[T]":
        return cls(encode_runs(seq))

    # -- access -------------------------------------------------------------

    @property
    def runs(self) -> List[Run[T]]:
        return list(self._runs)

    def to_iterable(self) -> List[T]:
        return decode_runs(self._runs)

    def __iter__(self) -> Iterator[T]:
        return iter_runs(self._runs)

    def __len__(self) -> int:
        return sum(r.freq for r in self._runs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceRLE):
            return NotImplemented
        return self._runs == other._runs

    def __repr__(self) -> str:
        return f"SequenceRLE({self._runs!r})"

    # -- operations ---------------------------------------------------------

    def concat(self, other: "SequenceRLE[T]") -> "SequenceRLE[T]":
        if not self._runs:
            return SequenceRLE(other._runs)
        if not other._runs:
            return SequenceRLE(self._runs)
        merged = list(self._runs)
        first = other._runs[0]
        if merged[-1].token == first.token:
            merged[-1] = Run(merged[-1].token, merged[-1].freq + first.freq)
            merged.extend(other._runs[1:])
        else:
            merged.extend(other._runs)
        return SequenceRLE(merged)

    # -- bit accounting -----------------------------------------------------

    def bit_size(self, token_width: int, freq_encoding: str = "leb128") -> int:
        """Bits used by this representation.

        token_width: fixed bit-width per token (caller picks based on the
            alphabet; e.g. 8 for bytes).
        freq_encoding: 'leb128' | 'gamma' | 'raw32'.
        """
        if freq_encoding not in ("leb128", "gamma", "raw32"):
            raise ValueError(f"unknown freq_encoding {freq_encoding!r}")

        total = 0
        for r in self._runs:
            total += token_width
            if freq_encoding == "leb128":
                total += leb128_bits(r.freq)
            elif freq_encoding == "gamma":
                total += gamma_bits(r.freq)
            else:  # raw32
                total += 32
        return total
