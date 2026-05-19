"""LZ77 sliding-window back-reference compression.

Captures **word-level** redundancy that RLE/BWT cannot: any sub-string that
has appeared in the recent window can be replaced with a (distance, length)
pointer. This is the same idea deflate / gzip / zstd / brotli are built on.

Output is four parallel streams:

  * ``controls``: 0 = literal, 1 = back-reference, per output token.
  * ``literals``: the literal bytes (only where ``controls`` == 0).
  * ``lengths``:  match length (only where ``controls`` == 1).
  * ``distances``: match distance (only where ``controls`` == 1).

Pure Python — slow for big inputs, fine for the benchmark sizes here.
Hash chain with a bounded walk; no lazy matching.
"""

from __future__ import annotations

from typing import List, Tuple

WINDOW = 32 * 1024
MIN_MATCH = 3
MAX_MATCH = 258
HASH_BITS = 15
MAX_CHAIN = 128


def lz77_encode(
    data: bytes,
    window: int = WINDOW,
    min_match: int = MIN_MATCH,
    max_match: int = MAX_MATCH,
    hash_bits: int = HASH_BITS,
    max_chain: int = MAX_CHAIN,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    n = len(data)
    if n == 0:
        return [], [], [], []
    table_size = 1 << hash_bits
    mask = table_size - 1
    head = [-1] * table_size
    prev = [-1] * n

    controls: List[int] = []
    literals: List[int] = []
    lengths: List[int] = []
    distances: List[int] = []

    def insert_hash(p: int) -> int:
        if p + min_match > n:
            return -1
        hv = ((data[p] << 16) | (data[p + 1] << 8) | data[p + 2]) & mask
        chain = head[hv]
        prev[p] = chain
        head[hv] = p
        return chain

    pos = 0
    while pos < n:
        chain_start = insert_hash(pos)

        best_len = 0
        best_dist = 0
        if chain_start >= 0:
            cand = chain_start
            count = 0
            limit = min(max_match, n - pos)
            target_first = data[pos : pos + min_match]
            while cand >= 0 and pos - cand <= window and count < max_chain:
                if data[cand : cand + min_match] == target_first:
                    # Extend match
                    m = min_match
                    while m < limit and data[cand + m] == data[pos + m]:
                        m += 1
                    if m > best_len:
                        best_len = m
                        best_dist = pos - cand
                        if m == max_match:
                            break
                cand = prev[cand]
                count += 1

        if best_len >= min_match:
            controls.append(1)
            lengths.append(best_len)
            distances.append(best_dist)
            # Insert hashes for the bytes we're skipping, so future matches
            # can still find them.
            for k in range(1, best_len):
                p = pos + k
                if p + min_match > n:
                    break
                insert_hash(p)
            pos += best_len
        else:
            controls.append(0)
            literals.append(data[pos])
            pos += 1

    return controls, literals, lengths, distances


def lz77_decode(
    controls: List[int],
    literals: List[int],
    lengths: List[int],
    distances: List[int],
) -> bytes:
    out = bytearray()
    lit_i = 0
    match_i = 0
    for c in controls:
        if c == 0:
            out.append(literals[lit_i])
            lit_i += 1
        else:
            length = lengths[match_i]
            dist = distances[match_i]
            match_i += 1
            start = len(out) - dist
            # Byte-by-byte copy: handles overlap (RLE-style repeat) correctly.
            for i in range(length):
                out.append(out[start + i])
    return bytes(out)
