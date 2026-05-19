"""Random-access compressed file format.

Block-based variant of ``vector_rle_shared``: the input is tokenized,
chunked into blocks of K tokens, and each block is independently
PPM-encoded against the shared base dictionary. A small index over
the blocks lets a reader decode only the block(s) needed to satisfy a
random byte- or word-position query.

Trade-off: each block re-pays PPM's model warmup cost (~5-10 % per
block depending on block size), so the compressed file is a few percent
larger than the non-random ``vector_rle_shared`` output. In exchange,
*any* byte or word position can be retrieved by decoding ~1/n of the
file.

File layout::

    leb128(uncompressed_len)
    leb128(base_dict_version)
    leb128(n_local_tokens)
    leb128(total_token_count)
    leb128(block_size_in_tokens)
    leb128(n_blocks)

    [if n_local_tokens > 0:]
        encode_ppm_stream(local_lengths_bytes)        # PPM-compressed
        encode_ppm_stream(concatenated_local_bytes)   # PPM-compressed

    block_byte_lengths[n_blocks]                      # leb128 each

    for each block:
        leb128(compressed_block_size)
        compressed_block_bytes                        # encode_ppm_seq output

Both the block-byte-length array (for byte-position lookups) and the
per-block compressed-size prefixes (for file-offset seeks) are simple
enough that the index is essentially free at the file-size level.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import List, Optional

from .bitpack import leb128_decode, leb128_encode
from .ppm import decode_ppm_seq, decode_ppm_stream, encode_ppm_seq, encode_ppm_stream
from .shared_dict import SharedDict
from .wordtok import detokenize, tokenize

BLOCK_SIZE_DEFAULT = 2048

_BASE_DICT_CACHE: Optional[SharedDict] = None


def _shared_base_dict() -> SharedDict:
    global _BASE_DICT_CACHE
    if _BASE_DICT_CACHE is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "data"
            / "base_english.dict"
        )
        _BASE_DICT_CACHE = SharedDict.load(path)
    return _BASE_DICT_CACHE


def _block_ppm_order(n_block_tokens: int) -> int:
    """PPM order for a *single* block. Smaller than the whole-file
    heuristic because each block restarts from scratch."""
    if n_block_tokens < 200:
        return 0
    if n_block_tokens < 1500:
        return 1
    return 2


def compress_random(data: bytes, block_size: int = BLOCK_SIZE_DEFAULT) -> bytes:
    """Compress with block-based random-access layout."""
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)

    base = _shared_base_dict()
    n_base = len(base)

    tokens = tokenize(data)
    n_tokens = len(tokens)

    # Partition tokens: in base vs novel.
    seen_local: set = set()
    novel_tokens: List[bytes] = []
    for t in tokens:
        if base.get_id(t) is None and t not in seen_local:
            seen_local.add(t)
            novel_tokens.append(t)
    novel_tokens.sort()
    n_local = len(novel_tokens)
    local_id_of = {t: n_base + i for i, t in enumerate(novel_tokens)}

    def to_id(t: bytes) -> int:
        bid = base.get_id(t)
        return bid if bid is not None else local_id_of[t]

    ids = [to_id(t) for t in tokens]
    n_blocks = (n_tokens + block_size - 1) // block_size

    out += leb128_encode(base.version)
    out += leb128_encode(n_local)
    out += leb128_encode(n_tokens)
    out += leb128_encode(block_size)
    out += leb128_encode(n_blocks)

    if n_local > 0:
        lengths_bytes = bytes(len(t) for t in novel_tokens)
        out += encode_ppm_stream(lengths_bytes, order=2)
        out += encode_ppm_stream(b"".join(novel_tokens), order=4)

    alphabet = n_base + n_local

    # Compute per-block byte lengths and compressed payloads.
    block_byte_lens: List[int] = []
    block_payloads: List[bytes] = []
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n_tokens)
        block_byte_len = sum(len(t) for t in tokens[start:end])
        block_byte_lens.append(block_byte_len)
        if alphabet >= 2:
            order = _block_ppm_order(end - start)
            payload = encode_ppm_seq(ids[start:end], alphabet, order=order)
        else:
            # Single-token alphabet: emit just the per-block token count.
            payload = leb128_encode(end - start)
        block_payloads.append(payload)

    # Write block byte lengths (the byte-offset index).
    for blen in block_byte_lens:
        out += leb128_encode(blen)

    # Write compressed blocks (each prefixed by its byte length).
    for payload in block_payloads:
        out += leb128_encode(len(payload))
        out += payload

    return bytes(out)


def decompress_random(blob: bytes) -> bytes:
    """Inverse of ``compress_random`` (full file)."""
    if not blob:
        return b""
    return RandomReader(blob).read_all()


# ---------------------------------------------------------------------------
# RandomReader — the headline feature: read into compressed bytes
# ---------------------------------------------------------------------------


class RandomReader:
    """Random-access view of a ``compress_random`` blob.

    Reads only the block(s) needed to satisfy a query. Suitable for
    workloads that touch a small fraction of the file (log tail, range
    scans, point lookups by token index).
    """

    __slots__ = (
        "_blob",
        "_input_len",
        "_n_tokens",
        "_block_size",
        "_n_blocks",
        "_base",
        "_novel",
        "_alphabet",
        "_block_byte_offsets",  # cumulative; len == n_blocks + 1
        "_block_file_offsets",  # absolute offsets into self._blob
    )

    def __init__(self, blob: bytes) -> None:
        if not blob:
            raise ValueError("Empty blob")
        self._blob = blob

        pos = 0
        self._input_len, pos = leb128_decode(blob, pos)
        if self._input_len == 0:
            self._n_tokens = 0
            self._n_blocks = 0
            self._block_size = 0
            self._base = _shared_base_dict()
            self._novel = []
            self._alphabet = 0
            self._block_byte_offsets = [0]
            self._block_file_offsets = []
            return

        self._base = _shared_base_dict()
        version, pos = leb128_decode(blob, pos)
        if version != self._base.version:
            raise ValueError(
                f"shared dict version mismatch: file v{version}, "
                f"runtime v{self._base.version}"
            )
        n_local, pos = leb128_decode(blob, pos)
        self._n_tokens, pos = leb128_decode(blob, pos)
        self._block_size, pos = leb128_decode(blob, pos)
        self._n_blocks, pos = leb128_decode(blob, pos)

        # Local delta dict.
        novel: List[bytes] = []
        if n_local > 0:
            lengths_bytes, pos = decode_ppm_stream(blob, pos)
            bytes_blob, pos = decode_ppm_stream(blob, pos)
            cursor = 0
            for L in lengths_bytes:
                novel.append(bytes_blob[cursor : cursor + L])
                cursor += L
        self._novel = novel
        self._alphabet = len(self._base) + n_local

        # Block byte-length array -> cumulative byte offsets.
        self._block_byte_offsets = [0]
        for _ in range(self._n_blocks):
            blen, pos = leb128_decode(blob, pos)
            self._block_byte_offsets.append(self._block_byte_offsets[-1] + blen)

        # Walk through the block-size prefixes to learn each block's
        # absolute file offset.
        self._block_file_offsets = []
        for _ in range(self._n_blocks):
            self._block_file_offsets.append(pos)
            clen, end_of_len = leb128_decode(blob, pos)
            pos = end_of_len + clen

    # -- introspection ---------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return self._input_len

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    # -- internals -------------------------------------------------------

    def _decode_block_ids(self, block_id: int) -> List[int]:
        pos = self._block_file_offsets[block_id]
        clen, pos = leb128_decode(self._blob, pos)
        payload = self._blob[pos : pos + clen]
        if self._alphabet <= 1:
            # Single-token alphabet: payload is just a token count.
            n, _ = leb128_decode(payload, 0)
            return [0] * n
        if not payload:
            return []
        ids, _ = decode_ppm_seq(payload)
        return ids

    def _id_to_token(self, bid: int) -> bytes:
        nb = len(self._base)
        if bid < nb:
            return self._base.get_token(bid)
        return self._novel[bid - nb]

    # -- random reads ----------------------------------------------------

    def read_word(self, k: int) -> bytes:
        """Return the k-th word token (0-indexed)."""
        if k < 0 or k >= self._n_tokens:
            raise IndexError(f"word index {k} out of range [0, {self._n_tokens})")
        block_id = k // self._block_size
        offset_in_block = k % self._block_size
        ids = self._decode_block_ids(block_id)
        return self._id_to_token(ids[offset_in_block])

    def read_byte_range(self, start: int, end: int) -> bytes:
        """Return uncompressed bytes ``[start, end)``."""
        if not (0 <= start <= end <= self._input_len):
            raise IndexError(
                f"byte range [{start}, {end}) invalid for total {self._input_len}"
            )
        if start == end:
            return b""

        # Find the first block whose byte range overlaps `start`.
        first_block = bisect.bisect_right(self._block_byte_offsets, start) - 1
        if first_block < 0:
            first_block = 0

        result = bytearray()
        block_id = first_block
        while block_id < self._n_blocks:
            block_byte_start = self._block_byte_offsets[block_id]
            if block_byte_start >= end:
                break
            ids = self._decode_block_ids(block_id)
            cursor = block_byte_start
            for bid in ids:
                tok = self._id_to_token(bid)
                tok_end = cursor + len(tok)
                if tok_end <= start:
                    cursor = tok_end
                    continue
                if cursor >= end:
                    break
                lo = max(start, cursor) - cursor
                hi = min(end, tok_end) - cursor
                result += tok[lo:hi]
                cursor = tok_end
                if cursor >= end:
                    break
            block_id += 1
        return bytes(result)

    def read_all(self) -> bytes:
        """Decompress the whole file. Equivalent to ``decompress_random``."""
        if self._n_tokens == 0:
            return b""
        out = bytearray()
        for b in range(self._n_blocks):
            ids = self._decode_block_ids(b)
            for bid in ids:
                out += self._id_to_token(bid)
        return bytes(out)
