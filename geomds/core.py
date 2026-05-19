from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


@dataclass
class Run(Generic[T]):
    token: T
    freq: int

    def __post_init__(self) -> None:
        if self.freq <= 0:
            raise ValueError(f"freq must be positive, got {self.freq}")
