# vrle — Vector Run-Length Encoding

A small Python library that reframes classic Run-Length Encoding as a
**vector** data structure: each run is a `(token, magnitude)` pair where the
**token is the identifier (direction)** and the **frequency of repetition is
the magnitude**. Storing each run once with a variable-length magnitude uses
fewer bits than the flat input on any data with redundancy, and casting runs
as vectors lets us do arithmetic on encoded streams without re-expanding them.

## Two forms

| Form           | Preserves order? | Lossless round-trip? | Supports vector math? |
|----------------|------------------|----------------------|-----------------------|
| `SequenceRLE`  | yes              | yes                  | no (it's a sequence)  |
| `MathRLE`      | no (sparse)      | no (counts only)     | yes (add, scale, dot, cosine) |

Both share the same internal representation (`Run(token, freq)`); they differ
in what you can do with them.

## Bit packing

Frequencies are stored with a variable-length encoding so small magnitudes
use few bits:

* **LEB128** — 7 value bits per byte, continuation bit in the MSB.
* **Elias gamma** — bit-level, optimal for small magnitudes (1 -> 1 bit,
  2..3 -> 3 bits, 4..7 -> 5 bits, ...).

The benchmark compares both against a fixed-width `(token, u32 freq)` baseline
and against the raw input.

## Quick start

```python
from vrle import SequenceRLE, MathRLE

# Sequence form: round-trips losslessly
s = SequenceRLE.from_iterable(b"aaaabbbcc")
print(s.runs)                      # [Run(97, 4), Run(98, 3), Run(99, 2)]
print(bytes(s.to_iterable()))      # b'aaaabbbcc'
print(s.bit_size(8, "gamma"))      # 31  (vs 72 raw)

# Mathematical form: arithmetic on encoded streams
a = MathRLE.from_iterable("the quick brown fox")
b = MathRLE.from_iterable("the lazy dog")
print(a.cosine_similarity(b))      # ~0.7
print((a + b).magnitudes)
```

## Run the benchmark

```
python -m vrle.bench
# or
python examples/compare.py
python examples/compare.py path/to/some/file
```

## Run the tests

```
pip install -e .[dev]
pytest -q
```

## Layout

```
vrle/
  core.py       Run dataclass + encode_runs / decode_runs / iter_runs
  sequence.py   SequenceRLE (ordered, lossless)
  mathvec.py    MathRLE (sparse vector with arithmetic)
  bitpack.py    LEB128 + Elias gamma + pack_runs / unpack_runs
  bench.py      bit-usage comparison harness
```
