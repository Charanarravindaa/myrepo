"""Move-To-Front transform.

Reversible; converts a stream so that recently-seen symbols become small
indices. Stacked before RLE/entropy coding, it converts "locality" into
"runs of small ints", which both RLE and entropy coders compress well.
"""

from __future__ import annotations


def mtf_encode(data: bytes, alphabet_size: int = 256) -> bytes:
    table = list(range(alphabet_size))
    out = bytearray(len(data))
    for i, b in enumerate(data):
        idx = table.index(b)
        out[i] = idx
        if idx:
            table.pop(idx)
            table.insert(0, b)
    return bytes(out)


def mtf_decode(data: bytes, alphabet_size: int = 256) -> bytes:
    table = list(range(alphabet_size))
    out = bytearray(len(data))
    for i, idx in enumerate(data):
        b = table[idx]
        out[i] = b
        if idx:
            table.pop(idx)
            table.insert(0, b)
    return bytes(out)
