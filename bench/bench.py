"""Performance comparison: geomds containers vs Python stdlib equivalents.

For each structure we run three workloads:
  * UNIFORM:  one token repeated N times (best case for run-coalescing)
  * BURSTY:   alternating runs of length R, K distinct tokens (realistic)
  * UNIQUE:   N distinct tokens (worst case; coalescing never fires)

We report wall time, peak resident memory (via tracemalloc on the build
phase), and the number of runs / nodes the geometric structure ended up
with.
"""
from __future__ import annotations

import gc
import heapq
import random
import sys
import time
import tracemalloc
from collections import Counter, deque
from typing import Callable, Tuple

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from geomds import (
    GeometricBitset,
    GeometricDeque,
    GeometricList,
    GeometricMultiset,
    GeometricPriorityQueue,
    GeometricQueue,
    GeometricStack,
)


N = 200_000


def time_and_mem(fn: Callable[[], object]) -> Tuple[float, int, object]:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, result


def workload(kind: str, n: int):
    if kind == "uniform":
        return ["x"] * n
    if kind == "bursty":
        K, R = 16, 64
        out = []
        for i in range(n // R + 1):
            tok = str(i % K)
            out.extend([tok] * R)
        return out[:n]
    if kind == "unique":
        return [str(i) for i in range(n)]
    raise ValueError(kind)


def fmt(t: float) -> str:
    if t < 1e-3:
        return f"{t*1e6:.0f}us"
    if t < 1:
        return f"{t*1e3:.1f}ms"
    return f"{t:.2f}s"


def fmt_mem(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    if b < 1024 * 1024:
        return f"{b/1024:.1f}KB"
    return f"{b/1024/1024:.2f}MB"


def header(title: str) -> None:
    print()
    print(f"=== {title} (n={N:,}) ===")
    print(f"{'workload':<10} {'impl':<22} {'time':>10} {'peak mem':>12} {'runs':>10}")


def row(workload_: str, impl: str, t: float, mem: int, runs: int | str) -> None:
    runs_str = str(runs) if isinstance(runs, int) else runs
    print(f"{workload_:<10} {impl:<22} {fmt(t):>10} {fmt_mem(mem):>12} {runs_str:>10}")


# ----------------------------------------------------------------------------
# Stack
# ----------------------------------------------------------------------------

def bench_stack():
    header("GeometricStack vs list (push N, then pop N)")
    for w in ("uniform", "bursty", "unique"):
        data = workload(w, N)

        def build_geo():
            s = GeometricStack[str]()
            for x in data:
                s.push(x)
            for _ in range(N):
                s.pop()
            return s

        def build_list():
            s: list[str] = []
            for x in data:
                s.append(x)
            for _ in range(N):
                s.pop()
            return s

        # Build geometric (also measure during fill, before pop)
        gs = GeometricStack[str]()
        for x in data:
            gs.push(x)
        runs = gs.runs()
        del gs

        t1, m1, _ = time_and_mem(build_geo)
        t2, m2, _ = time_and_mem(build_list)
        row(w, "GeometricStack", t1, m1, runs)
        row(w, "list", t2, m2, "-")

    # The asymptotic best case: push N copies of one token with push_many.
    def push_many():
        s = GeometricStack[str]()
        s.push_many("x", N)
        return s

    def push_loop_list():
        s: list[str] = []
        for _ in range(N):
            s.append("x")
        return s

    t1, m1, _ = time_and_mem(push_many)
    t2, m2, _ = time_and_mem(push_loop_list)
    row("uniform*", "GeometricStack.push_many", t1, m1, 1)
    row("uniform*", "list.append x N", t2, m2, "-")


# ----------------------------------------------------------------------------
# Queue
# ----------------------------------------------------------------------------

def bench_queue():
    header("GeometricQueue vs collections.deque (enqueue N, dequeue N)")
    for w in ("uniform", "bursty", "unique"):
        data = workload(w, N)

        def build_geo():
            q = GeometricQueue[str]()
            for x in data:
                q.enqueue(x)
            for _ in range(N):
                q.dequeue()
            return q

        def build_deque():
            q: deque[str] = deque()
            for x in data:
                q.append(x)
            for _ in range(N):
                q.popleft()
            return q

        gq = GeometricQueue[str]()
        for x in data:
            gq.enqueue(x)
        runs = gq.runs()
        del gq

        t1, m1, _ = time_and_mem(build_geo)
        t2, m2, _ = time_and_mem(build_deque)
        row(w, "GeometricQueue", t1, m1, runs)
        row(w, "collections.deque", t2, m2, "-")


# ----------------------------------------------------------------------------
# Deque
# ----------------------------------------------------------------------------

def bench_deque():
    header("GeometricDeque vs collections.deque (mixed push/pop both ends)")
    rng = random.Random(0)
    ops = [(rng.random() < 0.5, rng.random() < 0.5) for _ in range(N)]  # (front?, push?)
    data = workload("bursty", N)

    def build_geo():
        d = GeometricDeque[str]()
        for i, (front, push) in enumerate(ops):
            if push or len(d) == 0:
                if front:
                    d.push_front(data[i])
                else:
                    d.push_back(data[i])
            else:
                if front:
                    d.pop_front()
                else:
                    d.pop_back()
        return d

    def build_dq():
        d: deque[str] = deque()
        for i, (front, push) in enumerate(ops):
            if push or len(d) == 0:
                if front:
                    d.appendleft(data[i])
                else:
                    d.append(data[i])
            else:
                if front:
                    d.popleft()
                else:
                    d.pop()
        return d

    t1, m1, g = time_and_mem(build_geo)
    t2, m2, _ = time_and_mem(build_dq)
    row("mixed", "GeometricDeque", t1, m1, g.runs())
    row("mixed", "collections.deque", t2, m2, "-")


# ----------------------------------------------------------------------------
# List (splice-heavy)
# ----------------------------------------------------------------------------

def bench_list():
    header("GeometricList splice vs list concatenation (K chunks of L items)")
    K, L = 1000, N // 1000

    def build_geo():
        chunks = []
        for i in range(K):
            g = GeometricList[str]()
            g.append(str(i % 4), L)
            chunks.append(g)
        merged = GeometricList[str]()
        for c in chunks:
            merged.splice(c)
        return merged

    def build_list():
        chunks = []
        for i in range(K):
            chunks.append([str(i % 4)] * L)
        merged: list[str] = []
        for c in chunks:
            merged.extend(c)
        return merged

    t1, m1, g = time_and_mem(build_geo)
    t2, m2, _ = time_and_mem(build_list)
    row("k-splice", "GeometricList", t1, m1, g.runs())
    row("k-splice", "list.extend", t2, m2, "-")


# ----------------------------------------------------------------------------
# Multiset
# ----------------------------------------------------------------------------

def bench_multiset():
    header("GeometricMultiset vs collections.Counter")
    rng = random.Random(0)
    # Zipf-ish: 90% of the mass on 100 tokens, 10% on the long tail.
    data = []
    heads = [f"h{i}" for i in range(100)]
    tails = [f"t{i}" for i in range(N)]
    for _ in range(N):
        if rng.random() < 0.9:
            data.append(rng.choice(heads))
        else:
            data.append(rng.choice(tails))

    def build_geo():
        m = GeometricMultiset[str]()
        for x in data:
            m.add(x)
        return m

    def build_counter():
        c: Counter = Counter()
        for x in data:
            c[x] += 1
        return c

    t1, m1, mset = time_and_mem(build_geo)
    t2, m2, ctr = time_and_mem(build_counter)
    row("zipf", "GeometricMultiset", t1, m1, mset.distinct())
    row("zipf", "collections.Counter", t2, m2, len(ctr))

    # Vector op
    m_a = build_geo()
    m_b = build_geo()
    t3, _, _ = time_and_mem(lambda: m_a.cosine(m_b))
    row("cosine", "GeometricMultiset", t3, 0, "-")


# ----------------------------------------------------------------------------
# Priority queue (streaming top-K heavy-hitter)
# ----------------------------------------------------------------------------

def bench_pqueue():
    header("GeometricPriorityQueue vs heapq+Counter (streaming top-1)")
    rng = random.Random(0)
    data = []
    heads = [f"h{i}" for i in range(100)]
    tails = [f"t{i}" for i in range(N)]
    for _ in range(N):
        if rng.random() < 0.9:
            data.append(rng.choice(heads))
        else:
            data.append(rng.choice(tails))

    def build_geo():
        pq = GeometricPriorityQueue[str]()
        for x in data:
            pq.push(x)
        return pq.peek_max()

    def build_baseline():
        c: Counter = Counter()
        for x in data:
            c[x] += 1
        return c.most_common(1)[0]

    t1, m1, top = time_and_mem(build_geo)
    t2, m2, top2 = time_and_mem(build_baseline)
    assert top[0] == top2[0]
    row("zipf", "GeometricPriorityQueue", t1, m1, "-")
    row("zipf", "Counter.most_common(1)", t2, m2, "-")


# ----------------------------------------------------------------------------
# Bitset
# ----------------------------------------------------------------------------

def bench_bitset():
    header("GeometricBitset vs set[int] (membership + union)")
    # Dense workload: insert 0..N (one giant run).
    def build_geo_dense():
        b = GeometricBitset()
        for i in range(N):
            b.add(i)
        return b

    def build_set_dense():
        s: set[int] = set()
        for i in range(N):
            s.add(i)
        return s

    t1, m1, gb = time_and_mem(build_geo_dense)
    t2, m2, _ = time_and_mem(build_set_dense)
    row("dense", "GeometricBitset", t1, m1, gb.runs())
    row("dense", "set[int]", t2, m2, "-")

    # Sparse / random workload: 10% density over [0, N).
    rng = random.Random(0)
    points = sorted(rng.sample(range(N), N // 10))

    def build_geo_sparse():
        b = GeometricBitset()
        for p in points:
            b.add(p)
        return b

    def build_set_sparse():
        s: set[int] = set()
        for p in points:
            s.add(p)
        return s

    t1, m1, gb = time_and_mem(build_geo_sparse)
    t2, m2, _ = time_and_mem(build_set_sparse)
    row("sparse", "GeometricBitset", t1, m1, gb.runs())
    row("sparse", "set[int]", t2, m2, "-")

    # Range union.
    a_geo = GeometricBitset(range(0, N))
    b_geo = GeometricBitset(range(N // 2, 3 * N // 2))
    a_set = set(range(0, N))
    b_set = set(range(N // 2, 3 * N // 2))

    t1, _, _ = time_and_mem(lambda: a_geo | b_geo)
    t2, _, _ = time_and_mem(lambda: a_set | b_set)
    row("union", "GeometricBitset", t1, 0, "-")
    row("union", "set[int]", t2, 0, "-")


def main():
    bench_stack()
    bench_queue()
    bench_deque()
    bench_list()
    bench_multiset()
    bench_pqueue()
    bench_bitset()


if __name__ == "__main__":
    main()
