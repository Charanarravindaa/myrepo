from __future__ import annotations

import heapq
from typing import Dict, Generic, Hashable, List, Tuple, TypeVar

T = TypeVar("T", bound=Hashable)


class GeometricPriorityQueue(Generic[T]):
    """Magnitude is priority. push(token) adds 1 to that token's magnitude;
    peek_max / pop_max return the highest-magnitude token. Stale heap entries
    are skipped lazily."""

    __slots__ = ("_counts", "_heap", "_tiebreaker")

    def __init__(self) -> None:
        self._counts: Dict[T, int] = {}
        self._heap: List[Tuple[int, int, T]] = []
        self._tiebreaker = 0

    def push(self, token: T, count: int = 1) -> None:
        if count <= 0:
            return
        self._counts[token] = self._counts.get(token, 0) + count
        self._tiebreaker += 1
        heapq.heappush(
            self._heap, (-self._counts[token], self._tiebreaker, token)
        )

    def _drain_stale(self) -> None:
        while self._heap:
            neg, _, tok = self._heap[0]
            cur = self._counts.get(tok, 0)
            if -neg == cur and cur > 0:
                return
            heapq.heappop(self._heap)

    def peek_max(self) -> Tuple[T, int]:
        self._drain_stale()
        if not self._heap:
            raise IndexError("peek_max on empty GeometricPriorityQueue")
        neg, _, tok = self._heap[0]
        return tok, -neg

    def pop_max(self) -> Tuple[T, int]:
        """Remove the highest-magnitude token entirely; return (token, magnitude)."""
        self._drain_stale()
        if not self._heap:
            raise IndexError("pop_max from empty GeometricPriorityQueue")
        neg, _, tok = heapq.heappop(self._heap)
        cur = self._counts.pop(tok)
        assert -neg == cur
        return tok, cur

    def __len__(self) -> int:
        return sum(self._counts.values())

    def distinct(self) -> int:
        return len(self._counts)

    def get(self, token: T) -> int:
        return self._counts.get(token, 0)
