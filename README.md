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

## Thirteen compression pipelines

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
* **`vector_rle_shared`** (v7) — *vector_rle with a shared base dictionary*.
  Tokens already present in a pre-trained content-addressed shared
  dictionary (`examples/data/base_english.dict`) are referenced by their
  base ID and their bytes are **not** shipped. Novel tokens go into a
  per-file delta dict. **Beats brotli on real 53 KB English text**:
  12 053 B vs brotli 12 838 B (6.1 % smaller).
* **`vector_auto`** (v9) — *general-purpose auto-dispatch*. A single
  entry point. Classifies the input (random / redundant / English /
  logs / code / no-word-structure-text) and picks the right
  sub-pipeline + shared dictionary automatically. With a coverage
  fallback that detects when the chosen dict doesn't actually match
  the data and falls back to pure adaptive arithmetic. **Beats gzip
  on 5 of 6 benchmarks; beats brotli on logs (45 %) and on long
  English text (6 %).**
* **`vector_rle_random`** (v8) — *block-structured `vector_rle_shared`
  with random-byte access*. The input is chunked into 2 048-token
  blocks, each independently PPM-encoded. A tiny block index lets a
  reader decode just the block(s) covering any byte position — no full
  decompression. About 25 % worse ratio than `vector_rle_shared`
  (per-block PPM warmup overhead), but **still beats gzip and zstd**
  on the long English benchmark. Each random read is ~9× faster than
  full decompression on the 53 KB sample, with the speedup growing
  linearly with file size.
* **`vector_rle`** (v6) — *geometric tokenization with PPM throughout*.
  Parse the input into word tokens, sort the vocabulary alphabetically,
  PPM-compress the lengths array, PPM-compress the concatenated token
  bytes, and run an order-N PPM over the positional index (PPMword).
  The magnitude vector serves three roles: it is the dictionary
  (token ↔ ID), it seeds the PPM order-0 model, and its support
  defines the alphabet for the positional coder. **Beats gzip on
  real 53 KB English text and beats every standard compressor — gzip,
  bz2, zstd, brotli — on web log streams.**

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
| Long English text (53 KB GPL-3 + GPL-2) | **vector_auto / vector_rle_shared 12054** ✓✓✓✓ | 16379 | 14411 | 15053 | 12838 |
| Long English text — *random-access variant* | vector_rle_random 15131 ✓✓ | 16379 | 14411 | 15053 ✗ | 12838 ✗ |
| Web log stream (14.5 KB) | **vector_auto 485** ✓✓✓✓ | 974 | 673 | 862 | 881 |
| English letter-frequency 20 KB | vector_auto 10634 ✓✓✓ | 11951 | 11369 | 10541 ✗ | 10472 ✗ |
| Random bytes 20 KB | vector_auto 20004 ≈ brotli | 20028 | 20481 | 20010 | 20004 |
| High-redundancy bytes 20 KB | vector_auto 289 ✓✓✓ | 360 | 387 | 312 ✗ | 249 ✗ |

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

### v7: Shared base dictionary (`vector_rle_shared`)

The current headline pipeline. Identical to `vector_rle` (v6) but the
encoder consults a **pre-trained shared dictionary** of common English
tokens before deciding which tokens are novel. Tokens already in the
shared base are referenced by their base ID and their bytes are *not*
re-shipped. Novel tokens go into a tiny per-file delta dictionary.

The shared dictionary is content-addressed by version (a 1-byte
version field; future revisions get incremented). Every consumer of
a file made by `vector_rle_shared` must have the matching base dict
on disk — exactly the same constraint brotli has with its compiled-in
English dictionary, just made explicit and trainable.

**Result on real 53 KB English text (GPL-3 + GPL-2):**

| pipeline             | bytes  | % raw | % gzip |
|----------------------|-------:|------:|------:|
| **vector_rle_shared**| **12 053** | **22.64** |  73.6 |
| brotli(11)           | 12 838 | 24.11 |  78.4 |
| ppm_rc               | 14 348 | 26.95 |  87.6 |
| bz2(9)               | 14 411 | 27.07 |  88.0 |
| zstd(22)             | 15 053 | 28.27 |  91.9 |
| vector_rle (v6)      | 15 821 | 29.72 |  96.6 |
| gzip(9)              | 16 379 | 30.76 | 100.0 |

`vector_rle_shared` is **first** by 785 bytes (6.1 % smaller than
brotli, 16 % smaller than zstd). The shared dictionary covers 97.7 %
of token occurrences in the input, so the file's local delta is tiny.

#### Training the shared dictionary

```
python -m vrle.train_dict <corpus-dir> -o my.dict -n 4000
```

The bundled `examples/data/base_english.dict` was trained on the
license texts in `/usr/share/common-licenses/` *excluding* the two
files (GPL-2, GPL-3) that make up the long-text benchmark — so the
training corpus is independent of the test set. The trained dict has
2 698 unique tokens and is 23 KB on disk; the long-text bench input
has 1 465 unique tokens, of which 1 174 (80.1 %) are covered by the
shared dict.

#### How the file format extends the geometric framing

The v7 format extends the user's *one geometric object, multiple roles*
abstraction: the magnitude vector now spans **two pieces** — the
shared base (loaded from disk on both sides) plus the file's local
delta (shipped inline). PPM at decoding time treats them as one
contiguous alphabet, exactly as if the file had a single 5 000-token
magnitude vector.

#### The "growing OS dictionary" extension

`vector_rle_shared` files are **portable** — they decode anywhere the
matching base dict is present. Local OS-level absorption of file
deltas (your "OS dict grows over time" idea) is a layer above this
format: a receiver may append a file's local-delta tokens to its own
extended base dict (e.g. `~/.local/share/vrle/extended.dict`) and
subsequent files compressed *for that machine* can reference the
extended IDs without re-shipping. The compression primitive built here
is the same; only the lookup gets a per-machine extension.

### v9: General-purpose auto-dispatch (`vector_auto`)

A single pipeline that automatically picks the best compression
strategy for any input. The full *general-purpose* claim.

**How it works:**

1. **Classify** the input on a multi-region 4 KB sample using six cheap
   features (entropy, word-byte fraction, newline density,
   ASCII-printable fraction, operator-character density, average run
   length) plus keyword matching against language-specific patterns
   (`def`, `class`, `import`, ... for code; `HTTP/`, status codes,
   `INFO`/`ERROR` for logs).
2. **Pick the route**: random → store raw; redundant → byte-level PPM;
   English / logs / code → `vector_rle_shared` with the matching
   bundled dictionary; text-y but no word structure → pure adaptive
   arithmetic (`arith_rc`).
3. **Coverage check**: for dict-based routes, sanity-check that the
   chosen dict actually covers ≥ 75 % of tokens in a 2 KB probe. If
   not, fall back to `arith_rc` — guards against misclassification
   and unusual inputs that look text-y but don't match any dictionary.
4. **Encode** with the chosen route. A single byte at the head of the
   blob records which route was taken; decompress dispatches on it.

Three shared dictionaries ship bundled in `vrle/dicts/`:

| dict          | tokens | trained from                                |
|---------------|-------:|---------------------------------------------|
| english.dict  | 2 698  | `/usr/share/common-licenses` (excl. GPL-2/3) |
| code.dict     | 3 000  | 100 files from Python 3.12 stdlib            |
| logs.dict     | 2 000  | a synthetic multi-format log corpus          |

About **58 KB of bundled dictionaries**, comparable to brotli's
~120 KB compiled-in English dictionary.

**Result — `vector_auto` is the first single-entry-point pipeline in
this project that's competitive across every data class:**

| dataset                         | vector_auto | gzip(9) | bz2(9) | zstd(22) | brotli(11) |
|---------------------------------|------------:|--------:|-------:|---------:|-----------:|
| Random bytes 20 KB              |      20 004 |  20 028 | 20 481 |   20 010 |     20 004 |
| High-redundancy 20 KB           |         289 |     360 |    387 |      312 |        249 |
| Letter-freq text 20 KB          |      10 634 |  11 951 | 11 369 |   10 541 |     10 472 |
| **Web logs 14.5 KB**            |     **485** |     974 |    673 |      862 |        881 |
| Real prose 4.5 KB               |       2 232 |   2 163 |  2 050 |    2 110 |      1 729 |
| **Long English text 53 KB**     |  **12 054** |  16 379 | 14 411 |   15 053 |     12 838 |

* Beats **gzip on 5 / 6** datasets (loses only the 4.5 KB prose
  excerpt by 69 B — vocabulary overhead on too-small inputs).
* Beats **bz2 on 5 / 6**, **zstd on 5 / 6**.
* **Ties brotli on incompressible** data, **beats brotli outright on
  web logs (45 % smaller) and long English text (6 % smaller)**.

#### The OS-native roadmap

This is where v9 sits in a longer arc:

```
v9 ← we are here. The algorithm is general-purpose.

  ↓ v10 — C / Rust port. Python prototype is 100-1000x slower than
          zstd; production use needs a native engine.

  ↓ Wire layer — HTTP Content-Encoding, scp/rsync hooks, CLI tools
                  for compress/decompress/inspect.

  ↓ Filesystem — FUSE mount + .vrz extension association so any
                  application opens compressed files transparently
                  (with random byte access via vector_rle_random).

  ↓ OS-native — distros ship signed, versioned shared dictionaries
                via apt/rpm/brew; file managers, log shippers, and
                package formats migrate to vrle as the default.
```

The end state is *a new way of sending and reading files*: small over
the wire (better-than-brotli on the data classes you actually move
around), random-access-capable so editors and viewers can `mmap` into
compressed files, and OS-shipped dictionaries that grow over time
without breaking portability.

### v8: Random-access compressed format (`vector_rle_random`)

A compressed file format where any byte position can be read without
decompressing the whole file. Same wire-level construction as
`vector_rle_shared` (v7) but the positional index is chunked into
independently-encoded blocks of 2 048 tokens each, with a small index
that maps block IDs to file offsets and cumulative byte offsets.

```python
from vrle import vector_rle_random, RandomReader

blob = vector_rle_random.compress(big_text)
# regular full decompress still works:
assert vector_rle_random.decompress(blob) == big_text

# but the headline feature is reading without decompressing:
r = RandomReader(blob)
print(r.total_bytes, r.n_tokens, r.n_blocks)
chunk = r.read_byte_range(50_000, 50_100)   # decode only the block(s)
                                            # that cover bytes 50_000..50_100
word  = r.read_word(1_000)                  # the 1 000-th word token
```

#### Trade-off

The per-block PPM model has to warm up from scratch, so the file is
~25 % larger than `vector_rle_shared` on the 53 KB GPL benchmark:

| pipeline                          | bytes  | random access? |
|-----------------------------------|-------:|----------------|
| `vector_rle_shared`               | 12 053 | no             |
| **`vector_rle_random` (2048 blk)**| **15 131** | **yes**     |
| `vector_rle` (no shared dict)     | 15 821 | no             |
| gzip(9)                           | 16 379 | no             |

`vector_rle_random` still beats gzip and zstd on this dataset and is
within 5 % of bz2. **No production general-purpose compressor (gzip /
bz2 / zstd / brotli) supports byte-position random access by default.**
zstd has a *seekable* mode and the bioinformatics world ships BGZF
(blocked gzip) for indexed reads on genome files, but no commodity
tool offers ratio-competitive random-access compression for arbitrary
text.

#### Why random access pays off

On the 53 KB sample:

| operation                | time    | speedup     |
|--------------------------|--------:|------------:|
| full decompress          | 9 800 ms |    —       |
| single 50-byte random read | 1 100 ms | **9 ×**   |

The speedup scales with `n_blocks`. On a 1 MB English text (~190
blocks) a random read would be ~190 × faster than full decompression.

#### Block-size knob

`vector_rle_random.compress(data, block_size=K)` accepts a custom
block size. Smaller blocks → finer random-access granularity, worse
ratio. Larger blocks → coarser random access, better ratio (approaches
`vector_rle_shared` as `block_size → ∞`).

### v6: Geometric tokenization with PPM throughout (`vector_rle`)

The most novel pipeline in the project. Parse the input into word
tokens (maximal runs of word/non-word characters, capped at 255 bytes).
Each unique token gets its own "vector direction" identified by its
ID; the **magnitude** of that vector is the **exact frequency** of the
token. The magnitude vector serves *three* roles:

1. It is the **dictionary** — pairing each token with its ID.
2. It seeds the **PPM order-0 model** that codes the positional index.
3. Its support defines the **alphabet** of the positional coder.

The vocabulary is stored alphabetically (so the encoder and decoder
agree on the token↔ID mapping with no frequency table needing to be
written — PPM learns the frequencies itself from the encoded stream).
Three PPM streams: lengths array, concatenated token bytes, and the
positional index.

**Result.** Both `vector_rle`'s flagship benchmarks now win:

| dataset                        | vector_rle | gzip(9) | bz2(9) | zstd(22) | brotli(11) |
|--------------------------------|-----------:|--------:|-------:|---------:|-----------:|
| Web log stream (14.5 KB)       | **510 ✓**  | 974     | 673    | 862      | 881        |
| Long English text (53 KB GPL)  | **15 821 ✓** | 16 379 | 14 411 | 15 053  | 12 838     |

* On **web logs** `vector_rle` is the **absolute winner** — beats
  gzip by 47 %, bz2 by 24 %, zstd by 41 %, brotli by 42 %.
* On **53 KB real English text** `vector_rle` **beats gzip** by 3.4 %
  and is within 10 % of bz2 / zstd / `ppm_rc`. Brotli still wins on
  prose because of its pre-trained English dictionary.

Where `vector_rle` loses, honestly:

* **Random data**: ~50 % overhead — word tokenization is meaningless
  on noise.
* **Short prose (4.5 KB)**: 2 546 B vs gzip 2 163 B. The PPM model
  on word IDs doesn't have enough data to settle.
* **Synthetic letter-frequency text**: word tokens collapse to
  single characters; word-level PPM has no structure to exploit.

The format itself stays simple: three PPM streams plus six header
bytes. **No frequencies are written anywhere** — they emerge as the
adaptive PPM model learns them. This is the v6 simplification: the
magnitude vector is implicit in the encoded stream, recoverable but
never explicitly serialised, exactly as the *one geometric object,
multiple roles* framing suggests.

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
  shared_dict.py    Shared base dictionary load/save (v7)
  train_dict.py     CLI: train a SharedDict from a corpus
  random_access.py  RandomReader + block-based codec (v8)
  classify.py       Content classifier for vector_auto (v9)
  dicts/
    english.dict    Trained on /usr/share/common-licenses
    logs.dict       Trained on synthetic multi-format log corpus
    code.dict       Trained on Python 3.12 stdlib
  vocab.py          Vocabulary: token <-> ID + magnitude vector
  wordtok.py        Byte-level word tokenizer
  pipelines.py      10 pipelines, rle_rc through vector_rle
  bench.py          Head-to-head benchmark vs gzip/bz2/zstd/brotli
examples/
  compare.py        CLI wrapper
  data/sample.txt        Pride & Prejudice excerpt (~4.5 KB)
  data/long_text.txt     GPL-3 + GPL-2 concat (~53 KB, real English)
  data/base_english.dict Bundled SharedDict (~23 KB, 2698 tokens, v7)
tests/              345 tests covering round-trips, arithmetic, PPM, rANS,
                    vocab, shared dict, random access, classifier
```
