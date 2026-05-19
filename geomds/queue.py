from __future__ import annotations

from collections import deque
from typing import Deque, Generic, Hashable, Iterator, TypeVar

from .core import Run

T = TypeVar("T", bound=Hashable)


class GeometricQueue(Generic[T]):
    """FIFO queue with adjacent-run coalescing at the tail and run-decrement
    dequeue at the head. enqueue_many(x, n) is O(1)."""

    __slots__ = ("_runs", "_size")

    def __init__(self) -> None:
        self._runs: Deque[Run[T]] = deque()
        self._size = 0

    def enqueue(self, token: T) -> None:
        if self._runs and self._runs[-1].token == token:
            self._runs[-1].freq += 1
        else:
            self._runs.append(Run(token, 1))
        self._size += 1

    def enqueue_many(self, token: T, count: int) -> None:
        if count <= 0:
            return
        if self._runs and self._runs[-1].token == token:
            self._runs[-1].freq += count
        else:
            self._runs.append(Run(token, count))
        self._size += count

    def dequeue(self) -> T:
        if not self._runs:
            raise IndexError("dequeue from empty GeometricQueue")
        front = self._runs[0]
        token = front.token
        if front.freq == 1:
            self._runs.popleft()
        else:
            front.freq -= 1
        self._size -= 1
        return token

    def peek(self) -> T:
        if not self._runs:
            raise IndexError("peek on empty GeometricQueue")
        return self._runs[0].token

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
