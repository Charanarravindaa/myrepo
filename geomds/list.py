from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Iterator, Optional, TypeVar

from .core import Run

T = TypeVar("T", bound=Hashable)


@dataclass
class _Node(Generic[T]):
    run: Run[T]
    prev: Optional["_Node[T]"] = None
    next: Optional["_Node[T]"] = None


class GeometricList(Generic[T]):
    """Doubly-linked list of runs. Splice at a same-token boundary coalesces
    both sides into a single node in O(1)."""

    __slots__ = ("_head", "_tail", "_size", "_nodes")

    def __init__(self) -> None:
        self._head: Optional[_Node[T]] = None
        self._tail: Optional[_Node[T]] = None
        self._size = 0
        self._nodes = 0

    def append(self, token: T, count: int = 1) -> None:
        if count <= 0:
            return
        if self._tail and self._tail.run.token == token:
            self._tail.run.freq += count
        else:
            node: _Node[T] = _Node(Run(token, count), prev=self._tail)
            if self._tail:
                self._tail.next = node
            else:
                self._head = node
            self._tail = node
            self._nodes += 1
        self._size += count

    def prepend(self, token: T, count: int = 1) -> None:
        if count <= 0:
            return
        if self._head and self._head.run.token == token:
            self._head.run.freq += count
        else:
            node: _Node[T] = _Node(Run(token, count), next=self._head)
            if self._head:
                self._head.prev = node
            else:
                self._tail = node
            self._head = node
            self._nodes += 1
        self._size += count

    def splice(self, other: "GeometricList[T]") -> None:
        """Move all of other onto the tail of self in O(1) (with one possible
        merge at the join). other becomes empty."""
        if not other._head:
            return
        if not self._tail:
            self._head = other._head
            self._tail = other._tail
            self._size = other._size
            self._nodes = other._nodes
        elif self._tail.run.token == other._head.run.token:
            self._tail.run.freq += other._head.run.freq
            joiner = other._head.next
            if joiner is None:
                pass
            else:
                joiner.prev = self._tail
                self._tail.next = joiner
                self._tail = other._tail
            self._size += other._size
            self._nodes += other._nodes - 1
        else:
            self._tail.next = other._head
            other._head.prev = self._tail
            self._tail = other._tail
            self._size += other._size
            self._nodes += other._nodes
        other._head = None
        other._tail = None
        other._size = 0
        other._nodes = 0

    def __len__(self) -> int:
        return self._size

    def runs(self) -> int:
        return self._nodes

    def compression(self) -> float:
        return self._size / max(1, self._nodes)

    def __iter__(self) -> Iterator[T]:
        node = self._head
        while node:
            for _ in range(node.run.freq):
                yield node.run.token
            node = node.next
