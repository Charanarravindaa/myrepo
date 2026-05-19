from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Iterator, List, TypeVar

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class Run(Generic[T]):
    token: T
    freq: int

    def __post_init__(self) -> None:
        if self.freq <= 0:
            raise ValueError(f"Run frequency must be positive, got {self.freq}")


def encode_runs(seq: Iterable[T]) -> List[Run[T]]:
    runs: List[Run[T]] = []
    it = iter(seq)
    try:
        current = next(it)
    except StopIteration:
        return runs
    count = 1
    for tok in it:
        if tok == current:
            count += 1
        else:
            runs.append(Run(current, count))
            current = tok
            count = 1
    runs.append(Run(current, count))
    return runs


def decode_runs(runs: Iterable[Run[T]]) -> List[T]:
    out: List[T] = []
    for run in runs:
        out.extend([run.token] * run.freq)
    return out


def iter_runs(runs: Iterable[Run[T]]) -> Iterator[T]:
    for run in runs:
        for _ in range(run.freq):
            yield run.token
