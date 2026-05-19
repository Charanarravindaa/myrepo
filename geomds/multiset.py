from __future__ import annotations

import math
from typing import Dict, Generic, Hashable, Iterable, Iterator, TypeVar

T = TypeVar("T", bound=Hashable)


class GeometricMultiset(Generic[T]):
    """Order-free magnitude container. Token is the dimension, count is the
    magnitude. Supports multiset algebra and vector ops (dot, cosine)."""

    __slots__ = ("_counts",)

    def __init__(self, items: Iterable[T] | None = None) -> None:
        self._counts: Dict[T, int] = {}
        if items:
            for x in items:
                self.add(x)

    def add(self, token: T, count: int = 1) -> None:
        if count <= 0:
            return
        self._counts[token] = self._counts.get(token, 0) + count

    def remove(self, token: T, count: int = 1) -> None:
        if count <= 0:
            return
        cur = self._counts.get(token, 0)
        if cur <= count:
            self._counts.pop(token, None)
        else:
            self._counts[token] = cur - count

    def __contains__(self, token: object) -> bool:
        return token in self._counts

    def __getitem__(self, token: T) -> int:
        return self._counts.get(token, 0)

    def __len__(self) -> int:
        return sum(self._counts.values())

    def distinct(self) -> int:
        return len(self._counts)

    def __iter__(self) -> Iterator[T]:
        for tok, n in self._counts.items():
            for _ in range(n):
                yield tok

    def __or__(self, other: "GeometricMultiset[T]") -> "GeometricMultiset[T]":
        out: GeometricMultiset[T] = GeometricMultiset()
        keys = set(self._counts) | set(other._counts)
        out._counts = {
            t: max(self._counts.get(t, 0), other._counts.get(t, 0)) for t in keys
        }
        return out

    def __and__(self, other: "GeometricMultiset[T]") -> "GeometricMultiset[T]":
        out: GeometricMultiset[T] = GeometricMultiset()
        for t, c in self._counts.items():
            o = other._counts.get(t, 0)
            if o:
                out._counts[t] = min(c, o)
        return out

    def __add__(self, other: "GeometricMultiset[T]") -> "GeometricMultiset[T]":
        out: GeometricMultiset[T] = GeometricMultiset()
        out._counts = dict(self._counts)
        for t, c in other._counts.items():
            out._counts[t] = out._counts.get(t, 0) + c
        return out

    def __sub__(self, other: "GeometricMultiset[T]") -> "GeometricMultiset[T]":
        out: GeometricMultiset[T] = GeometricMultiset()
        for t, c in self._counts.items():
            d = c - other._counts.get(t, 0)
            if d > 0:
                out._counts[t] = d
        return out

    def dot(self, other: "GeometricMultiset[T]") -> int:
        if len(self._counts) > len(other._counts):
            return other.dot(self)
        return sum(c * other._counts.get(t, 0) for t, c in self._counts.items())

    def cosine(self, other: "GeometricMultiset[T]") -> float:
        d = self.dot(other)
        na = math.sqrt(sum(c * c for c in self._counts.values()))
        nb = math.sqrt(sum(c * c for c in other._counts.values()))
        if not na or not nb:
            return 0.0
        return d / (na * nb)
