# geomds — Geometric Data Structures

A small Python package that takes the `(token, magnitude)` pair from the
Vector-RLE compression work and surfaces it as a family of general-purpose
container ADTs. The unifying property is **automatic adjacent-run
coalescing**: pushing the same token onto a same-token boundary mutates a
counter in-place instead of allocating a new node.

## Structures

| Container | Backing | Coalesces at |
|---|---|---|
| `GeometricStack` | `list[Run]` | top |
| `GeometricQueue` | `deque[Run]` | tail |
| `GeometricDeque` | `deque[Run]` | both ends |
| `GeometricList` | doubly-linked `Run` nodes | any splice boundary |
| `GeometricMultiset` | `dict[token, count]` | by token (order-free) |
| `GeometricPriorityQueue` | `dict + lazy max-heap` | magnitude = priority |
| `GeometricBitset` | sorted `[start, length]` runs | adjacent / overlapping runs |

All seven share the same `Run` primitive (`geomds/core.py`). 34 unit tests
cover correctness; see `tests/`.

## Performance

Benchmark on Python 3.11.15, n = 200 000. Run with `python3 bench/bench.py`.
Three workloads test the spectrum: `uniform` is one token repeated (best
case for coalescing), `bursty` is alternating runs of length 64 across 16
tokens (realistic), `unique` is all-distinct tokens (worst case — coalescing
never fires).

### Where geomds clearly wins

| Op | geomds | stdlib | factor |
|---|---|---|---|
| Stack: `push_many("x", N)` vs `list.append` × N | 51 µs / 850 B | 59 ms / 1.55 MB | **~1000× faster, ~1800× less mem** |
| Stack: uniform push-then-pop, **memory** | 896 B | 1.55 MB | **~1700× less mem** |
| Bitset: `add(i) for i in range(N)`, **memory** | 368 B | 16.79 MB | **~45 000× less mem** |
| Bitset: range union (1 run each) | 63 µs | 5.7 ms (`set\|set`) | **~90× faster** |
| Multiset on Zipf input: `add` × N | 74 ms | 163 ms (`Counter`) | **~2.2× faster** |
| Multiset: cosine similarity of two N-token streams | 3.3 ms | (would need re-expansion) | new capability |

### Where stdlib wins

| Op | geomds | stdlib | factor |
|---|---|---|---|
| Stack: uniform per-element push | 475 ms | 77 ms | **6× slower** |
| Stack: unique workload (200k distinct) | 768 ms / 18.3 MB | 77 ms / 1.55 MB | **10× slower, 12× more mem** |
| Queue: unique workload | 624 ms / 18.4 MB | 80 ms / 1.57 MB | **8× slower, 12× more mem** |
| Bitset: sparse random adds (~17 986 runs) | 362 ms | 1.2 ms (`set`) | **300× slower** |
| PriorityQueue: streaming top-1 over Zipf | 829 ms / 25.7 MB | 143 ms / 612 KB (`Counter.most_common`) | **6× slower, 40× more mem** |

### Honest read

- **Memory wins are real and large** whenever the data has runs. Uniform
  stacks and dense bitsets are essentially free.
- **Time wins require explicit bulk ops** (`push_many`, range union,
  pre-aggregated workloads). Per-element pure-Python push is dominated by
  attribute access and dataclass field reads versus the C-optimized
  `list`/`deque`/`set`/`dict`.
- **Worst case (no runs) costs ~2× memory and ~8–10× time** — every element
  becomes its own `Run` object with a freq of 1.
- **`GeometricPriorityQueue` underperforms** because the lazy-heap pattern
  fills with stale entries on every push (one per token-magnitude
  transition). A Fibonacci-heap or in-place decrease-key implementation
  would close the gap; the simple lazy version trades correctness for
  speed.

### When to actually use these

- **Editor undo stacks / animation timelines / kernel log buffers** —
  `GeometricStack` or `GeometricQueue` with `push_many` and explicit run
  semantics. Burst coalescing for free.
- **Dense or range-structured integer sets** — `GeometricBitset`. This is
  the same shape Roaring Bitmaps uses in ClickHouse / Druid / Lucene; the
  big union/intersection speedup is the proof.
- **Bag-of-words vector ops without re-expansion** — `GeometricMultiset.cosine`.
- **Anywhere you splice many like-token segments together** —
  `GeometricList.splice`, which merges in O(1) at same-token boundaries.

### When to skip them

- Hot path with all-distinct elements (logs of UUIDs, raw byte streams of
  random data). Use `list` / `deque` / `set`.
- Streaming top-K where you actually need fast pops. Use `heapq` + `Counter`.
- Any workload that does a lot of random index access — none of these
  structures give you O(1) `[i]`; the best you can do is O(log runs) over
  cumulative magnitudes, which isn't implemented yet.

## Layout

```
geomds/
  core.py          Run dataclass
  stack.py         GeometricStack
  queue.py         GeometricQueue
  deque.py         GeometricDeque
  list.py          GeometricList
  multiset.py      GeometricMultiset
  pqueue.py        GeometricPriorityQueue
  bitset.py        GeometricBitset
tests/             34 unit tests, all passing
bench/bench.py     Comparative benchmark
```

Run `python3 -m pytest tests/ -q` and `python3 bench/bench.py`.
