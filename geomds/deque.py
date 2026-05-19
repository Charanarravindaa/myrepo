from __future__ import annotations

from collections import deque
from typing import Deque, Generic, Hashable, Iterator, TypeVar

from .core import Run

T = TypeVar("T", bound=Hashable)


class GeometricDeque(Generic[T]):
    """Double-ended geometric container. Coalesces at both ends."""

    __slots__ = ("_runs", "_size")

    def __init__(self) -> None:
        self._runs: Deque[Run[T]] = deque()
        self._size = 0

    def push_back(self, token: T, count: int = 1) -> None:
        if count <= 0:
            return
        if self._runs and self._runs[-1].token == token:
            self._runs[-1].freq += count
        else:
            self._runs.append(Run(token, count))
        self._size += count

    def push_front(self, token: T, count: int = 1) -> None:
        if count <= 0:
            return
        if self._runs and self._runs[0].token == token:
            self._runs[0].freq += count
        else:
            self._runs.appendleft(Run(token, count))
        self._size += count

    def pop_back(self) -> T:
        if not self._runs:
            raise IndexError("pop_back from empty GeometricDeque")
        back = self._runs[-1]
        token = back.token
        if back.freq == 1:
            self._runs.pop()
        else:
            back.freq -= 1
        self._size -= 1
        return token

    def pop_front(self) -> T:
        if not self._runs:
            raise IndexError("pop_front from empty GeometricDeque")
        front = self._runs[0]
        token = front.token
        if front.freq == 1:
            self._runs.popleft()
        else:
            front.freq -= 1
        self._size -= 1
        return token

    def peek_front(self) -> T:
        if not self._runs:
            raise IndexError("peek_front on empty GeometricDeque")
        return self._runs[0].token

    def peek_back(self) -> T:
        if not self._runs:
            raise IndexError("peek_back on empty GeometricDeque")
        return self._runs[-1].token

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
