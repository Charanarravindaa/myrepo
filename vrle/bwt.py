"""Burrows-Wheeler Transform.

Reversible permutation that clusters identical symbols together. The
classic step that makes bzip2 beat gzip on text: after BWT, similar
contexts produce runs of the same byte, which MTF + RLE + entropy coding
then squeezes very efficiently.

Implementation is the naive O(n^2 log n) suffix-array-via-sort. Fine for
the ~100 KB benchmark inputs; would need a real suffix-array algorithm
(SA-IS) to scale further.
"""

from __future__ import annotations

from typing import Tuple


def bwt_encode(data: bytes) -> Tuple[bytes, int]:
    """Return (transformed, primary_index). primary_index == -1 for empty."""
    n = len(data)
    if n == 0:
        return b"", -1
    # Sort suffix indices by their cyclic rotation. We compare via
    # data*2 slicing which is O(n^2 log n) in the worst case but cheap
    # in practice on short inputs because Python's sort dispatches to
    # C-level memcmp on bytes slices.
    doubled = data + data
    order = sorted(range(n), key=lambda i: doubled[i : i + n])
    primary = order.index(0)
    # Last column = data[(idx - 1) mod n] for each idx in order.
    transformed = bytes(data[(i - 1) % n] for i in order)
    return transformed, primary


def bwt_decode(transformed: bytes, primary_index: int) -> bytes:
    """Invert bwt_encode using the standard LF-mapping."""
    n = len(transformed)
    if n == 0:
        return b""
    # First column = sorted last column.
    counts = [0] * 256
    for b in transformed:
        counts[b] += 1
    # starts[b] = first row in F whose first byte is b
    starts = [0] * 256
    total = 0
    for b in range(256):
        starts[b] = total
        total += counts[b]
    # rank[i] = how many times transformed[i] has occurred up to and
    # including position i (1-based count within that symbol).
    next_index = [0] * n
    seen = [0] * 256
    for i, b in enumerate(transformed):
        next_index[i] = starts[b] + seen[b]
        seen[b] += 1
    # Walk in reverse: out is read backwards.
    out = bytearray(n)
    row = primary_index
    for k in range(n - 1, -1, -1):
        out[k] = transformed[row]
        row = next_index[row]
    return bytes(out)
