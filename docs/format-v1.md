# `.vrz` file format v1 — design specification

**Status:** design draft, pre-implementation. Targets the C/Rust port
of `vrle`. The Python prototype today serves as the algorithmic
reference; this document defines the *operational* layer (streaming,
integrity, versioning, memory bounds, threading, ABI) that the native
port must implement.

**Scope:** this document specifies the on-disk format, the streaming
protocol, error-recovery semantics, the C ABI for the codec, and the
implementation roadmap. The algorithmic core (PPM, word tokenisation,
shared dictionaries, classifier, random-access blocks) is unchanged
from v9 of the Python prototype.

## 1. Goals

1. **OS-native operability.** A C ABI suitable for libc / kernel
   pipelines, FUSE mounts, package managers, and language bindings.
2. **Streaming.** Incremental encode + decode that runs in
   bounded memory regardless of total input/output size. Required
   for pipes, sockets, and any context where the file is larger than
   RAM.
3. **Random access.** Read any byte range without decompressing the
   whole file (the v8 feature, now made first-class).
4. **Error containment.** A single corrupted byte must not invalidate
   the rest of the file. Standard expectation for any storage codec.
5. **Versioned dictionaries.** Files reference a content-addressed
   dict ID (SHA-256/128). Receivers either have the matching dict or
   reject the file deterministically. Dicts are OS-distributed
   (apt/rpm/brew packages).
6. **Forward / backward compatibility.** Major-version bumps allowed
   to break compatibility; minor-version bumps must remain
   forward-compatible (old readers see new files and either decode
   or fail cleanly, never silently produce wrong bytes).

## 2. Non-goals

These are deliberately *out of scope* for v1:

- **Hot-path swap/zram compression.** Even the best C/Rust port
  won't match LZ4's 1+ GB/s for swap. `vrle` is a cold-storage and
  interactive-file format, not a swap-page codec.
- **Encryption.** Block-level CRC for integrity is in scope. Encryption
  belongs at a higher layer (LUKS / fscrypt / TLS), not in the codec.
- **Erasure coding.** Out of scope; pairs with separate FEC layers if
  needed.
- **Beating PPMd by a wide margin.** Within ~6 % on text is acceptable.
  The differentiator is the framework, not the ratio.

## 3. File layout

All multi-byte integers are little-endian. All offsets are
zero-indexed bytes from the start of the file.

### 3.1 Fixed header (64 bytes)

```
offset  size  field              description
   0     4    magic              "VRLE"  (0x56 0x52 0x4C 0x45)
   4     1    version_major      1
   5     1    version_minor      0
   6     2    feature_flags      bitfield (see §3.5)
   8     1    profile_tag        codec/route tag (see §3.6)
   9     1    block_layout       0 = streaming-only, 1 = random-access
  10     8    uncompressed_size  bytes; UINT64_MAX = unknown (streaming)
  18    16    dict_id            SHA-256/128 of the shared dict; 0 = none
  34     4    block_size         max uncompressed bytes per block
  38     8    n_blocks           total blocks; 0 = unknown (streaming)
  46     8    footer_offset      absolute offset of the footer;
                                 0 if no footer (pure streaming)
  54     8    reserved           must be zero
  62     2    header_crc32       CRC32C of bytes 0..61
```

The 64-byte size is chosen so a single `read(64)` populates everything
the decoder needs to dispatch.

### 3.2 Blocks (variable)

Immediately after the header, blocks are written sequentially. Each
block:

```
varint(N)        compressed payload length in bytes (LEB128 unsigned)
4 bytes          CRC32C of the payload bytes that follow
N bytes          compressed payload
```

The payload format is determined by `profile_tag` in the header (see
§3.6). For random-access mode, blocks are *independently* decodable
(each block carries enough state to decode standalone). For
streaming-only mode, blocks share state and must be decoded
sequentially.

### 3.3 Footer (random-access mode only)

When `block_layout = 1`, after the last block the file contains:

```
8 bytes * n_blocks   absolute file offsets of each block's varint-N
                     start (i.e. the first byte of the block record)
8 bytes              cumulative uncompressed byte offset after block 0,
                     block 1, ..., block n_blocks-1
                     (n_blocks * 8 bytes)
4 bytes              footer_crc32 (CRC32C of preceding footer bytes)
8 bytes              footer_offset (must match the header's footer_offset)
```

The trailing `footer_offset` lets a seekable reader open the file,
seek to `(file_size - 8)`, read the footer offset, then read the
footer directly without scanning blocks. Critical for fast random
opens.

### 3.4 Streaming-only files

When `block_layout = 0`:

- `n_blocks` in the header may be `0` (unknown at start).
- `uncompressed_size` may be `UINT64_MAX` (unknown).
- `footer_offset = 0`.
- No footer is written.
- The end-of-stream is signalled by a sentinel block: `varint(0)`
  CRC=0 (a zero-length payload). Decoder stops on this.

A streaming-only file can be made random-access-friendly later by
running `vrle reindex` (offline) which walks blocks once and appends
a footer.

### 3.5 Feature flags

```
bit 0  has_random_access     footer present, blocks independent
bit 1  has_block_crc         each block payload protected by CRC32C
bit 2  has_dict              dict_id non-zero, decoder must have dict
bit 3  has_file_hash         file_hash field present in footer
bit 4  uses_multithreading_hint   blocks were encoded in parallel; receiver MAY decode in parallel
bit 5-15  reserved (must be zero)
```

Major-version compatibility rule: a reader that doesn't understand a
set flag MUST refuse to decode (don't silently produce wrong bytes).
Minor-version evolution is allowed only via flags that the old reader
*can* safely ignore (e.g. an optional metadata block referenced by a
new reserved flag).

### 3.6 Profile tag

The profile tag identifies which codec produced the payload. Matches
the routing tags already used by `vector_auto`:

```
0   ppm_byte           byte-level PPM-C order 4
1   vrle_nodict        word-level vrle, no shared dict
2   vrle_dict_eng      word-level vrle, english.dict
3   vrle_dict_logs     word-level vrle, logs.dict
4   vrle_dict_code     word-level vrle, code.dict
5   arith_byte         pure adaptive arithmetic order 0
6   store_raw          payload is the raw bytes (incompressible)
7-127  reserved for future codecs
128-255  reserved for vendor/OS extensions
```

When `dict_id != 0`, the profile_tag MUST be in {2, 3, 4} (or whatever
future tags reference a dict). When `dict_id == 0`, the profile_tag
MUST be in the dict-free set {0, 1, 5, 6}.

## 4. Streaming protocol

### 4.1 Encoder state machine

```
INIT  → write header  → BLOCK_OPEN
BLOCK_OPEN  → buffer input until block_size bytes accumulated  → BLOCK_FULL
BLOCK_FULL  → compress buffer, emit varint(N), CRC, payload    → BLOCK_OPEN
BLOCK_OPEN  → finish() called  → flush partial block, emit sentinel, [maybe footer], DONE
```

Encoder API (logical, language-agnostic):

```
new(profile, dict_id, block_size, mode) → encoder
feed(encoder, in_bytes) → out_bytes (may be empty)
finish(encoder)         → out_bytes (header may be rewritten)
free(encoder)
```

In streaming mode (`block_layout = 0`), `finish()` only emits the
sentinel block; the header `uncompressed_size` / `n_blocks` stay as
"unknown" sentinels.

In random-access mode, `finish()` must rewrite the header's
`uncompressed_size`, `n_blocks`, `footer_offset` *and* emit the
footer. This requires the output sink be seekable. For non-seekable
sinks (sockets, pipes), random-access mode is invalid — implementations
MUST refuse it and require either:

- caller buffers the whole output and patches the header at the end, or
- caller uses streaming-only mode.

### 4.2 Decoder state machine

```
INIT  → read 64-byte header  → HEADER_OK (or HEADER_BAD)
HEADER_OK  → read varint(N) + CRC + payload → decompress payload → BLOCK_OPEN
BLOCK_OPEN → emit decoded bytes
BLOCK_OPEN → on next varint(0), DONE
```

Decoder API:

```
new(dict_resolver_callback) → decoder
feed(decoder, in_bytes)     → decoded_out_bytes
header(decoder)             → parsed header struct or NULL if not yet seen
finish(decoder)             → final decoded bytes; checks expected size matches header
free(decoder)
```

The `dict_resolver_callback` is invoked once when the header's
`dict_id` is non-zero. It returns either:

- the in-memory dict (caller has it loaded), or
- an error code (caller doesn't have it; decoder errors out
  with `ERR_MISSING_DICT` and the dict_id so the caller can fetch
  and retry).

### 4.3 Bounded memory

The decoder MUST run in O(`block_size` + `context_tree_budget`)
memory regardless of input size. Encoder same bound. Implementations
MUST honour a configurable `context_tree_budget` (default 64 MB).

When the PPM context tree hits the budget, the encoder triggers a
**context tree rebuild**:

1. Halve all counts (this also happens at MAX_TOTAL — same path).
2. Drop any context whose total count rounds to 0.
3. Emit a block boundary if not already at one.

The decoder mirrors deterministically by reading a 1-bit flag in the
block header (currently in the upper bit of the varint payload-length
prefix). Same rebuild on both sides keeps state synchronised.

## 5. Random-access read protocol

```
open(path):
  read file_size
  seek(file_size - 8); read footer_offset
  seek(footer_offset); read entire footer; verify footer_crc32
  build the in-memory index (block_id → file_offset, cum_byte_offset)

read_byte_range(start, end):
  first_block = bisect_right(cum_byte_offsets, start) - 1
  decode_block(first_block); copy bytes [start..end) from its output
  decode_block(first_block + 1); copy [start..end) from its output
  ...until cum_byte_offsets[k] >= end
```

For random-access mode, every block must be independently decodable.
That means each block's encoder state initialises fresh — same shared
dict + same profile + same parameters, but no context state inherited
from prior blocks. This is the ~3-5 % ratio cost we already accept.

For "warm" random access (sequential scans, prefetched reads), the
decoder MAY cache the last-decoded block's output bytes.

## 6. Error containment

Every block carries a CRC32C over its compressed payload. On
checksum mismatch:

- **Streaming decode**: the decoder enters `RECOVERING` state, emits
  an error event (with byte offset + block index), then attempts to
  resync by scanning forward for the next valid block boundary. If
  it finds one within `recovery_limit` bytes (default 64 KB),
  decoding resumes. Otherwise the decoder errors out cleanly.
- **Random-access read**: the affected block returns `ERR_CRC`. All
  other blocks remain readable.

Header CRC mismatch is fatal — the decoder refuses to operate without
a known-good header.

### 6.1 Recovery markers

Optional: every K-th block (default K=64) may include a 4-byte
recovery marker `0xVR0xLEC0xAB` (TBD) immediately before the varint.
If the resync scanner finds this marker, it knows the next byte is a
block-length varint with high probability. This is *belt-and-braces*
over the CRC-based resync.

## 7. Multi-threading

Random-access mode is embarrassingly parallel:

- **Parallel encode**: split the input into `n_threads` equal-byte
  ranges. Each thread independently encodes its range as a sequence
  of blocks. Concatenate. Rewrite the header + emit the footer
  on the main thread. Block boundaries respect thread boundaries
  (no block straddles two threads' input ranges).
- **Parallel decode**: trivially parallel — each block is independent.

Streaming-only mode cannot parallelise encode (state dependency
between blocks). It CAN parallelise decode if the decoder is given a
seekable input — same trick as random-access decode, since blocks are
still self-contained at the wire level (each has its own length
prefix + CRC).

The `uses_multithreading_hint` flag advises the receiver that the
producer parallelised; the receiver MAY parallelise decode but doesn't
have to.

## 8. Versioning rules

- **Same major version**: any reader that supports `version_major = X`
  must successfully decode any file with that major version, EXCEPT
  when an unsupported feature flag is set, in which case it must
  refuse with `ERR_UNSUPPORTED_FEATURE`.
- **Different major version**: reader refuses with `ERR_VERSION`.
- **Reserved bits / fields**: encoders MUST zero them. Decoders MUST
  refuse if any reserved bit is non-zero (defends against future
  ambiguity).
- **Dict version**: separate from file version. Dicts are
  content-addressed; an OS may have multiple dicts installed
  (e.g. `english.v1`, `english.v2`). The file's `dict_id` selects
  which one.

## 9. Cryptographic integrity

### 9.1 Dict identifier

```
dict_id = SHA-256(dict_serialised_bytes)[0:16]
```

Truncating to 128 bits keeps the header compact while keeping
collision probability negligible across realistic dictionary
populations (~2^64 dicts before a 50% collision chance).

### 9.2 Optional file content hash

If `has_file_hash` (bit 3) is set, the footer includes an extra
32-byte field after the offset table:

```
file_content_sha256   32 bytes   SHA-256 of all uncompressed bytes
```

This lets recipients verify end-to-end integrity (the per-block CRCs
catch flips on disk, but a content hash catches encoder bugs and
deliberate tampering when paired with detached signatures).

### 9.3 Dict authentication

Out of scope for the codec. OS package managers sign the
`/usr/share/vrle/dicts/*.dict` files via existing infrastructure (apt
signatures, rpm signatures). The codec checks SHA-256 match between
file's `dict_id` and the disk dict; signature validation is the
package manager's job.

## 10. C ABI

```c
#include <stdint.h>
#include <stddef.h>

/* opaque handles */
typedef struct vrle_encoder_t vrle_encoder_t;
typedef struct vrle_decoder_t vrle_decoder_t;
typedef struct vrle_dict_t    vrle_dict_t;

/* error codes */
typedef enum {
    VRLE_OK = 0,
    VRLE_ERR_VERSION,
    VRLE_ERR_UNSUPPORTED_FEATURE,
    VRLE_ERR_BAD_MAGIC,
    VRLE_ERR_HEADER_CRC,
    VRLE_ERR_BLOCK_CRC,
    VRLE_ERR_MISSING_DICT,
    VRLE_ERR_TRUNCATED,
    VRLE_ERR_OOM,
    VRLE_ERR_INVALID_ARG,
    VRLE_ERR_NOT_SEEKABLE,    /* random-access mode on non-seekable sink */
    VRLE_ERR_INTERNAL,
} vrle_status_t;

/* dict ops */
vrle_status_t vrle_dict_load(const char* path, vrle_dict_t** out);
void          vrle_dict_free(vrle_dict_t*);
void          vrle_dict_id(const vrle_dict_t*, uint8_t out_id[16]);

/* encoder */
typedef struct {
    uint8_t  profile_tag;     /* 0xFF = "auto" — encoder picks */
    uint8_t  block_layout;    /* 0 = streaming, 1 = random-access */
    uint32_t block_size;      /* 0 = default (64 KB) */
    uint32_t context_budget;  /* PPM memory budget in MB; 0 = default 64 */
    const vrle_dict_t* dict;  /* may be NULL */
    uint8_t  flags;           /* bitmask: VRLE_ENC_F_FILE_HASH, ... */
} vrle_encoder_params_t;

vrle_status_t vrle_enc_new(const vrle_encoder_params_t*, vrle_encoder_t** out);
void          vrle_enc_free(vrle_encoder_t*);

/* feed input; writes 0..out_cap bytes to out_buf. *written = bytes written. */
vrle_status_t vrle_enc_feed(
    vrle_encoder_t*,
    const uint8_t* in, size_t in_len,
    uint8_t* out_buf, size_t out_cap, size_t* written
);

/* finalise; may write final bytes + (if random-access) trigger header
   rewrite via a separate seekable_sink hook the caller installed. */
vrle_status_t vrle_enc_finish(
    vrle_encoder_t*,
    uint8_t* out_buf, size_t out_cap, size_t* written
);

/* decoder */
typedef vrle_status_t (*vrle_dict_resolver_t)(
    const uint8_t dict_id[16], const vrle_dict_t** out, void* userdata
);

vrle_status_t vrle_dec_new(
    vrle_dict_resolver_t resolver, void* resolver_userdata,
    vrle_decoder_t** out
);
void          vrle_dec_free(vrle_decoder_t*);

vrle_status_t vrle_dec_feed(
    vrle_decoder_t*,
    const uint8_t* in, size_t in_len,
    uint8_t* out_buf, size_t out_cap, size_t* written
);

/* random-access reader, opened against a file path */
typedef struct vrle_reader_t vrle_reader_t;
vrle_status_t vrle_reader_open(const char* path,
                               vrle_dict_resolver_t resolver, void* userdata,
                               vrle_reader_t** out);
uint64_t      vrle_reader_total_bytes(const vrle_reader_t*);
uint64_t      vrle_reader_n_blocks(const vrle_reader_t*);
vrle_status_t vrle_reader_range(vrle_reader_t*,
                                uint64_t start, uint64_t end,
                                uint8_t* out_buf, size_t out_cap, size_t* written);
void          vrle_reader_close(vrle_reader_t*);
```

Memory ownership: the codec OWNS its internal buffers; the caller owns
the `out_buf` it provides. No allocations happen outside `vrle_*_new`
in the steady state — feed/finish only write into caller buffers.

Threadsafety: each handle is single-threaded. Multiple encoder
handles can coexist on different threads. The dict is shareable
read-only across threads (immutable after `vrle_dict_load`).

## 11. Python wrapper

The C ABI is wrapped via `cffi` (decision: easier than ctypes for
struct-heavy ABIs, faster than pure-Python wrappers). The existing
Python API stays unchanged at the user level:

```python
from vrle import vector_auto, RandomReader, StreamEncoder, StreamDecoder

# unchanged from v9:
blob = vector_auto.compress(data)
out  = vector_auto.decompress(blob)

# new in v10 (the streaming layer):
enc = StreamEncoder(profile='auto')
for chunk in input_chunks:
    sys.stdout.buffer.write(enc.feed(chunk))
sys.stdout.buffer.write(enc.finish())

# random access is the existing API:
r = RandomReader.open(path)
print(r.total_bytes, r.n_blocks)
chunk = r.read_byte_range(50_000, 50_100)
```

Under the hood, `vector_auto.compress` and `RandomReader` dispatch to
the native C implementation when the `_vrle` extension is loaded, and
fall back to pure Python (with the existing Cython acceleration for
the PPM hot path) otherwise.

## 12. Implementation roadmap

Ordered, with rough effort estimates assuming one focused engineer
who already knows the Python prototype:

1. **Header / footer / block envelope (week 1)** — pure plumbing.
   Implement the C-level header read/write, varint helpers, CRC32C,
   footer assembly. Validate by round-tripping fixture files generated
   from the Python prototype.
2. **PPM core in C (weeks 2-4)** — port `_encode_loop` /
   `_decode_loop` from `vrle/ppm.py` with sparse-context storage
   (replace per-context `list[α+1]` with a hash map of seen symbols).
   This alone closes the ~13× throughput gap.
3. **Streaming API (week 5)** — wrap the PPM core with the
   block-encoder state machine + CRC + sentinel block.
4. **Dict loader + `dict_id` plumbing (week 5)** — read existing
   `.dict` files, compute SHA-256/128, expose via `vrle_dict_t`.
5. **Random-access reader (week 6)** — port the footer-driven reader
   from `vrle/random_access.py`.
6. **Multi-threaded encoder (week 7)** — split-by-thread, sequential
   header rewrite.
7. **Test corpus parity (week 8)** — run the same bench as the Python
   prototype. Verify ratio is unchanged within ±1 % per file
   (bit-exact would be nice but rounding in arithmetic-coder integer
   math may differ slightly across implementations).
8. **lzbench codec entry (week 9)** — wrap `vrle_compress_buffer` /
   `vrle_decompress_buffer` as an lzbench codec. Submit upstream PR.
9. **FUSE filesystem (week 10+)** — `vrlefs` mounts a directory;
   files inside are stored as `.vrz` with transparent read/write.

Total realistic budget for v1: **~10 weeks of focused work** for one
engineer. Half that with two engineers working in parallel (PPM core
and stream/random-access layers are independent until step 7).

## 13. Open questions

- **CRC choice**: CRC32C (Castagnoli) chosen for hardware
  acceleration availability on x86 (`crc32` instructions) and arm64
  (`+crc` extension). Alternative: BLAKE3 for fewer collisions at
  similar cost. Decision: ship CRC32C for v1; consider BLAKE3 if a
  real-world hash collision is observed.
- **Block size default**: 64 KB. Larger → better ratio, worse random
  access granularity. Tunable per-file.
- **Endianness**: little-endian everywhere. No big-endian support
  needed for v1.
- **Memory budget default**: 64 MB. Same order as zstd / brotli
  high-quality levels.
- **Context tree GC algorithm**: simple halve-and-prune on budget
  hit. More sophisticated (LRU, frequency-based) is a v2 concern.

## 14. What changes in the Python prototype before the port

A few alignments are worth doing while the prototype is still pure
Python, so the port has a clean target:

1. **Add CRC32C to existing block format** in `vrle/random_access.py`.
   Currently we have no per-block integrity check; the port shouldn't
   add this as a "feature" — it should already be there.
2. **Refactor `compress_random` to write the v1 header layout** so
   existing tests already exercise the spec'd header.
3. **Wire `dict_id` (SHA-256/128) into `SharedDict`** in
   `vrle/shared_dict.py`. The existing `version: int` field becomes a
   legacy compatibility marker; the SHA-256-derived `dict_id` is the
   canonical reference.
4. **Add a `StreamEncoder` / `StreamDecoder` wrapper around the
   existing buffer-based pipelines** as a pure-Python implementation
   of the streaming API. This proves the streaming semantics work
   before we re-implement them in C.

These four changes are ~1-2 days of work in Python and convert the
prototype into a "v1 reference implementation" the port can mirror
byte-for-byte.
