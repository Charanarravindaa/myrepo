from __future__ import annotations

from typing import Generic, Hashable, Iterator, List, TypeVar

from .core import Run

T = TypeVar("T", bound=Hashable)


class GeometricStack(Generic[T]):
    """LIFO stack with adjacent-run coalescing.

    push(x) onto a top whose token equals x is O(1) increment instead of a new
    allocation. push_many(x, n) is O(1) regardless of n. Random index access
    is O(log runs) via cumulative magnitudes (not implemented here; use the
    list view if you need it).
    """

    __slots__ = ("_runs", "_size")

    def __init__(self) -> None:
        self._runs: List[Run[T]] = []
        self._size = 0

    def push(self, token: T) -> None:
        if self._runs and self._runs[-1].token == token:
            self._runs[-1].freq += 1
        else:
            self._runs.append(Run(token, 1))
        self._size += 1

    def push_many(self, token: T, count: int) -> None:
        if count <= 0:
            return
        if self._runs and self._runs[-1].token == token:
            self._runs[-1].freq += count
        else:
            self._runs.append(Run(token, count))
        self._size += count

    def pop(self) -> T:
        if not self._runs:
            raise IndexError("pop from empty GeometricStack")
        top = self._runs[-1]
        token = top.token
        if top.freq == 1:
            self._runs.pop()
        else:
            top.freq -= 1
        self._size -= 1
        return token

    def peek(self) -> T:
        if not self._runs:
            raise IndexError("peek on empty GeometricStack")
        return self._runs[-1].token

    def top_run(self) -> Run[T] | None:
        return self._runs[-1] if self._runs else None

    def __len__(self) -> int:
        return self._size

    def runs(self) -> int:
        return len(self._runs)

    def compression(self) -> float:
        return self._size / max(1, len(self._runs))

    def __iter__(self) -> Iterator[T]:
        for run in self._runs:
            for _ in range(run.freq):
                yield run.token

    def clear(self) -> None:
        self._runs.clear()
        self._size = 0
