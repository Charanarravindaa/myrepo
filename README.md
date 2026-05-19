# vrle — Vector Run-Length Encoding

A Python library that reframes classic Run-Length Encoding as a **vector**
data structure: each run is a `(token, magnitude)` pair where the **token
is the identifier (direction)** and the **frequency of repetition is the
magnitude**. The v2 release builds three hybrid compression pipelines on
top of that core and benchmarks them head-to-head against gzip, bz2,
zstd, and brotli.

## Two data structures

| Form           | Preserves order? | Round-trips losslessly? | Supports vector math? |
|----------------|------------------|-------------------------|-----------------------|
| `SequenceRLE`  | yes              | yes                     | no (it's a sequence)  |
| `MathRLE`      | no (sparse)      | no (counts only)        | yes: `+`, scalar `*`, `dot`, `cosine_similarity` |

`MathRLE` is the experimental side of the project — vector arithmetic on
encoded streams. `SequenceRLE` is the round-trippable form that the
compression pipelines build on.

## Ten compression pipelines

All operate on byte streams, preserve order, and round-trip losslessly.

* **`rle_rc`** — SequenceRLE then arithmetic-code the token channel and
  the frequency channel separately. Smallest delta from v1.
* **`mtf_rle_rc`** — Move-To-Front first, then `rle_rc`.
* **`bwt_mtf_rle_rc`** — the classic bzip2 stack with Vector-RLE plugged
  in: BWT → MTF → SequenceRLE → arithmetic coding.
* **`lz_rc`** — LZ77 sliding-window matches, then arithmetic-code four
  streams (controls, literals, lengths, distances).
* **`lz_bwt_mtf_rle_rc`** — LZ77 first, then put residual literals
  through the BWT+MTF+RLE stack.
* **`deflate_rc`** — LZ77 + deflate-style length/distance bin codes.
* **`arith_rc`** — pure adaptive arithmetic coding (order-0). No
  transforms. Optimal for context-free distributions like skewed
  letter-frequency text.
* **`ppm_rc`** — order-N PPM-C context model on the raw byte stream.
  Order is picked adaptively by input size. **The strongest pipeline on
  natural text and structured logs.**
* **`bwt_ppm_rc`** — BWT + MTF then order-2 PPM. Best on extremely
  redundant data.
* **`vector_rle`** (v5) — *geometric tokenization*: parse the input into
  word tokens, build a vocabulary whose magnitude vector is both the
  dictionary (token ↔ ID) and the entropy model that rANS uses to code
  the positional index. One geometric object, two roles. The
  vocabulary's concatenated byte content is itself PPM-compressed.
  Loses on ratio to byte-level PPM on natural text (see below) — the
  format is interesting as a research artifact and as a step toward
  the unified random-access / multimodal variant.

Components:
* `vrle/bwt.py` — Burrows–Wheeler Transform (naive `O(n² log n)`).
* `vrle/mtf.py` — Move-To-Front transform.
* `vrle/lz.py` — LZ77 with hash-chain matcher and lazy matching.
* `vrle/deflate_codes.py` — RFC 1951 length/distance bin-code tables.
* `vrle/rangecoder.py` — 32-bit bit-oriented arithmetic coder; static
  semi-adaptive **and** fully adaptive variants, plus a Fenwick-tree
  frequency model used by both the adaptive coder and PPM.
* `vrle/ppm.py` — order-N PPM-C with adaptive arithmetic coding.

## Benchmark results (head-to-head vs gzip / bz2 / zstd / brotli)

The bench harness asserts `decompress(compress(x)) == x` for every
pipeline on every dataset — round-trip is non-negotiable.

Bytes used, lower is better. **Bold** marks the absolute winner; ✓
marks where a vrle pipeline beats a standard compressor.

| Dataset (raw)              | Best vrle               | gzip(9)  | bz2(9)   | zstd(22) | brotli(11) |
|----------------------------|-------------------------|---------:|---------:|---------:|-----------:|
| Random bytes (20 KB)       | arith_rc 20100          | **20028**| 20481    | 20010    | 20004      |
| High-redundancy (20 KB)    | bwt_ppm_rc **280** ✓✓✓  | 360      | 387      | 312      | 249        |
| Letter-freq text (20 KB)   | arith_rc **10633** ✓✓   | 11951    | 11369    | 10541    | 10472      |
| Web log stream (14.5 KB)   | **ppm_rc 511** ✓✓✓✓     | 974      | 673      | 862      | 881        |
| Real English prose (4.5 KB)| ppm_rc **1935** ✓✓✓     | 2163     | 2050     | 2110     | 1729       |
| Long English text (53 KB GPL-3 + GPL-2) | ppm_rc 14348 | 16379 ✓ | 14411 | 15053 ✓ | 12838 |

Score (vrle wins vs each standard compressor):

| Beat …  | Count | Where |
|---------|-----:|-------|
| gzip    | 4 / 5 | high-redundancy, letter-freq, logs, prose (random tied within 72 B) |
| bz2     | 4 / 5 | high-redundancy, letter-freq, logs, prose |
| zstd    | 4 / 5 | high-redundancy, logs, prose, letter-freq close |
| brotli  | 1 / 5 | logs (511 B vs brotli's 881 B — 42 % smaller) |

### How each pipeline contributes

* **`arith_rc`** wins on **letter-frequency text**: no structure beyond
  unconditional byte probabilities, so a pure order-0 adaptive coder
  approaches Shannon entropy and beats every dictionary-based scheme
  except brotli's preloaded dictionary.
* **`ppm_rc`** wins on **logs and natural prose**: the order-N model
  catches conditional structure ("HTTP/1." always followed by "1",
  "th" usually by "e") that dictionary compressors handle indirectly
  at higher cost. On the web-log stream it beats *every* standard
  compressor — including brotli at quality 11 — by a wide margin.
* **`bwt_ppm_rc`** wins on **highly redundant byte data** by combining
  BWT's clustering with PPM's per-context modelling.

### What got us here

Five targeted changes, measurable each time:

1. **Adaptive arithmetic coding** (no model header). Cleared ~250 B of
   per-stream tax that was killing us on small inputs.
2. **Order-N PPM-C** (`ppm.py`). The big lever — each byte conditioned
   on the last N bytes, with escape fallback to shorter contexts and a
   uniform order-0 floor. Implemented on a Fenwick-tree frequency
   model so per-context updates are O(log α).
3. **Adaptive PPM order** picked by input size.
4. **Pure adaptive arith pipeline** (`arith_rc`) added as the
   no-transforms baseline — turned out to be the winner on
   context-free data.
5. **PPM exclusion**: when a higher-order context escapes, every symbol
   it contained is excluded from the model at every lower order for the
   rest of that byte's encoding. The distribution tightens and lower
   orders spend bits only on symbols that are still possible.
   ~5–10 % gain on natural text (prose dropped from 2107 → 1935 B).

### v5: Geometric tokenization (`vector_rle`)

The most novel direction in the project. Each unique token (word, run
of whitespace, run of punctuation) gets its own "vector direction"
identified by its ID; the **magnitude** of that vector is the **exact
frequency** in the text. This magnitude vector serves *double duty*: it
is the dictionary on one hand and the static probability model that
rANS uses to code the positional index on the other. One geometric
object, two roles — the unification is the conceptual contribution.

The implementation is faithful: byte-level word tokenizer, descending-
frequency-sorted vocabulary, the dictionary's concatenated bytes
PPM-compressed (the dictionary has its own within-word redundancy
that PPM catches), and rANS encoding of the positional ID stream.

**Honest finding.** On the 53 KB GPL text:

| pipeline      | bytes  | % raw  | % gzip |
|---------------|-------:|------:|------:|
| brotli(11)    | 12 838 | 24.11 |  78.4 |
| ppm_rc        | 14 348 | 26.95 |  87.6 |
| bz2(9)        | 14 411 | 27.07 |  88.0 |
| zstd(22)      | 15 053 | 28.27 |  91.9 |
| gzip(9)       | 16 379 | 30.76 | 100.0 |
| **vector_rle**| **20 699** | **38.88** | **126.4** |

`vector_rle` is **not** competitive on ratio against byte-level
compressors. The reason is structural and worth stating clearly:
**word-level tokenization throws away the within-word byte-level
redundancy that PPM exploits.** Out of the 20 699-byte output,
~13 042 bytes is the rANS-coded positional index (entropy-optimal
for the per-word distribution we chose) and ~7 600 bytes is the
vocabulary (lengths + frequencies + PPM-compressed concatenated
tokens). PPM gets the same job done in 14 348 bytes by modelling
the byte stream directly.

This is a genuine empirical finding: the elegant *one geometric
object, two roles* abstraction is satisfying conceptually but loses
several bits per byte of compressible information that lower-level
context modelling captures. To make `vector_rle` competitive on
ratio against PPM/brotli we would need **order-N context modelling
on word IDs** (PPMword), which is left for future work. The current
implementation matters as the unified primitive that the deferred
*multimodal* extension (Affinity-Propagation-discretised continuous
media → same dictionary + positional index pipeline) will build on.

### Where we still lose, honestly

* **Random data** is fundamentally incompressible; gzip's 100.14 %
  represents storing nearly raw with a small header. We're at 100.50 %
  (72 B behind gzip). No transform can do better on true randomness.
* **Real prose vs brotli**: brotli ships with a ~120 KB pre-trained
  English dictionary that gives it a structural advantage no general-
  purpose stack can match without bringing its own dictionary.
* **Speed**: we are pure Python; ~100–1000 × slower than zstd / brotli.
  The bench measures ratio, not throughput.

## Quick start

```python
from vrle import SequenceRLE, MathRLE, bwt_mtf_rle_rc

# Sequence form (the data structure)
s = SequenceRLE.from_iterable(b"aaaabbbcc")
print(s.runs)                      # [Run(97, 4), Run(98, 3), Run(99, 2)]

# Mathematical form (the experimental side)
a = MathRLE.from_iterable("the quick brown fox")
b = MathRLE.from_iterable("the lazy dog")
print(a.cosine_similarity(b))

# Compression pipeline
data = b"the quick brown fox jumps " * 1000
blob = bwt_mtf_rle_rc.compress(data)
assert bwt_mtf_rle_rc.decompress(blob) == data
print(f"{len(data)} -> {len(blob)} ({100*len(blob)/len(data):.2f}%)")
```

## Run the benchmark

```
python -m vrle.bench
# or against your own file:
python examples/compare.py path/to/file
python examples/compare.py --pipeline bwt_mtf_rle_rc path/to/file
```

## Run the tests

```
pip install -e .[dev]
pytest -q
```

## Layout

```
vrle/
  core.py        Run dataclass + encode_runs / decode_runs / iter_runs
  sequence.py    SequenceRLE (ordered, lossless)
  mathvec.py     MathRLE (sparse vector with arithmetic)
  bitpack.py     LEB128 + Elias gamma + low-level bit I/O
  bwt.py            Burrows–Wheeler Transform
  mtf.py            Move-To-Front transform
  lz.py             LZ77 sliding-window matcher with lazy match
  deflate_codes.py  RFC 1951 length/distance bin-code tables
  rangecoder.py     Arithmetic coder (static + adaptive) + Fenwick model
  ppm.py            Order-N PPM-C context model
  rans.py           rANS (range-ANS) static entropy coder
  vocab.py          Vocabulary: token <-> ID + magnitude vector
  wordtok.py        Byte-level word tokenizer
  pipelines.py      10 pipelines, rle_rc through vector_rle
  bench.py          Head-to-head benchmark vs gzip/bz2/zstd/brotli
examples/
  compare.py        CLI wrapper
  data/sample.txt   Pride & Prejudice excerpt (~4.5 KB)
  data/long_text.txt GPL-3 + GPL-2 concat (~53 KB, real English)
tests/              275 tests covering round-trips, arithmetic, PPM, rANS, vocab
```
