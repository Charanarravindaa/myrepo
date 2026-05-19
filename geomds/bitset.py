from __future__ import annotations

from typing import Iterable, Iterator, List


class GeometricBitset:
    """Set of non-negative integers stored as runs of consecutive values.
    Adjacent and overlapping runs are coalesced. Union/intersection are
    linear in the number of runs, not in the cardinality."""

    __slots__ = ("_runs",)

    def __init__(self, items: Iterable[int] = ()) -> None:
        self._runs: List[List[int]] = []
        for x in items:
            self.add(x)

    def _bisect_runs(self, n: int) -> int:
        """First index i with _runs[i][0] > n."""
        lo, hi = 0, len(self._runs)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._runs[mid][0] <= n:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def add(self, n: int) -> None:
        if n < 0:
            raise ValueError("negative integers not supported")
        runs = self._runs
        idx = self._bisect_runs(n)
        if idx > 0:
            r = runs[idx - 1]
            if r[0] <= n < r[0] + r[1]:
                return
            if r[0] + r[1] == n:
                r[1] += 1
                if idx < len(runs) and runs[idx][0] == r[0] + r[1]:
                    r[1] += runs[idx][1]
                    runs.pop(idx)
                return
        if idx < len(runs) and runs[idx][0] == n + 1:
            runs[idx][0] = n
            runs[idx][1] += 1
            return
        runs.insert(idx, [n, 1])

    def discard(self, n: int) -> None:
        runs = self._runs
        idx = self._bisect_runs(n)
        if idx == 0:
            return
        r = runs[idx - 1]
        if not (r[0] <= n < r[0] + r[1]):
            return
        if r[1] == 1:
            runs.pop(idx - 1)
        elif n == r[0]:
            r[0] += 1
            r[1] -= 1
        elif n == r[0] + r[1] - 1:
            r[1] -= 1
        else:
            left_len = n - r[0]
            right_start = n + 1
            right_len = r[0] + r[1] - right_start
            r[1] = left_len
            runs.insert(idx, [right_start, right_len])

    def __contains__(self, n: object) -> bool:
        if not isinstance(n, int):
            return False
        idx = self._bisect_runs(n)
        if idx == 0:
            return False
        r = self._runs[idx - 1]
        return r[0] <= n < r[0] + r[1]

    def __len__(self) -> int:
        return sum(r[1] for r in self._runs)

    def runs(self) -> int:
        return len(self._runs)

    def compression(self) -> float:
        n = len(self)
        return n / max(1, len(self._runs))

    def __iter__(self) -> Iterator[int]:
        for start, length in self._runs:
            yield from range(start, start + length)

    def __or__(self, other: "GeometricBitset") -> "GeometricBitset":
        out = GeometricBitset()
        merged: List[List[int]] = []
        i = j = 0
        a, b = self._runs, other._runs
        while i < len(a) and j < len(b):
            if a[i][0] <= b[j][0]:
                merged.append(list(a[i])); i += 1
            else:
                merged.append(list(b[j])); j += 1
        while i < len(a):
            merged.append(list(a[i])); i += 1
        while j < len(b):
            merged.append(list(b[j])); j += 1
        coalesced: List[List[int]] = []
        for r in merged:
            if coalesced and coalesced[-1][0] + coalesced[-1][1] >= r[0]:
                last = coalesced[-1]
                last[1] = max(last[0] + last[1], r[0] + r[1]) - last[0]
            else:
                coalesced.append(r)
        out._runs = coalesced
        return out

    def __and__(self, other: "GeometricBitset") -> "GeometricBitset":
        out = GeometricBitset()
        result: List[List[int]] = []
        i = j = 0
        a, b = self._runs, other._runs
        while i < len(a) and j < len(b):
            ax, al = a[i]
            bx, bl = b[j]
            ae, be = ax + al, bx + bl
            lo, hi = max(ax, bx), min(ae, be)
            if lo < hi:
                result.append([lo, hi - lo])
            if ae <= be:
                i += 1
            else:
                j += 1
        out._runs = result
        return out
