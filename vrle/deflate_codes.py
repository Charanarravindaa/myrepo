"""Deflate-style bin codes for LZ77 lengths and distances.

The trick that makes deflate / gzip beat naive LZ77 encodings:

  * 29 **length codes** covering lengths 3..258. Each code spans a
    range of lengths with a fixed number of extra raw bits. The code
    goes through the entropy coder (cheap, since the distribution is
    skewed toward small codes); the extra bits are emitted raw (uniform
    inside the bin).
  * 30 **distance codes** covering distances 1..32768 in doubling bins.
    Same idea — code is entropy-coded, in-bin offset is raw.

Total: about 1 byte per match pair after entropy coding, vs ~1.5–2 bytes
when each value is emitted as LEB128 then arithmetic-coded.

Tables are from RFC 1951 §3.2.5, renumbered to start at 0.
"""

from __future__ import annotations

from typing import List, Tuple

# (base_length, extra_bits) for codes 0..28 (corresponding to deflate's 257..285)
LENGTH_CODES: List[Tuple[int, int]] = [
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 1),
    (13, 1),
    (15, 1),
    (17, 1),
    (19, 2),
    (23, 2),
    (27, 2),
    (31, 2),
    (35, 3),
    (43, 3),
    (51, 3),
    (59, 3),
    (67, 4),
    (83, 4),
    (99, 4),
    (115, 4),
    (131, 5),
    (163, 5),
    (195, 5),
    (227, 5),
    (258, 0),  # special: length 258 only
]

# (base_distance, extra_bits) for codes 0..29
DISTANCE_CODES: List[Tuple[int, int]] = [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 1),
    (7, 1),
    (9, 2),
    (13, 2),
    (17, 3),
    (25, 3),
    (33, 4),
    (49, 4),
    (65, 5),
    (97, 5),
    (129, 6),
    (193, 6),
    (257, 7),
    (385, 7),
    (513, 8),
    (769, 8),
    (1025, 9),
    (1537, 9),
    (2049, 10),
    (3073, 10),
    (4097, 11),
    (6145, 11),
    (8193, 12),
    (12289, 12),
    (16385, 13),
    (24577, 13),
]

MAX_LENGTH = 258
MAX_DISTANCE = 32768
N_LENGTH_CODES = len(LENGTH_CODES)
N_DISTANCE_CODES = len(DISTANCE_CODES)


def _build_length_lookup():
    L_to_code = [-1] * (MAX_LENGTH + 1)
    L_to_extra = [0] * (MAX_LENGTH + 1)
    for code in range(N_LENGTH_CODES - 1):  # codes 0..27
        base = LENGTH_CODES[code][0]
        next_base = LENGTH_CODES[code + 1][0]
        for L in range(base, next_base):
            if L > MAX_LENGTH:
                break
            L_to_code[L] = code
            L_to_extra[L] = L - base
    L_to_code[258] = 28
    L_to_extra[258] = 0
    return L_to_code, L_to_extra


def _build_distance_lookup():
    D_to_code = [-1] * (MAX_DISTANCE + 1)
    D_to_extra = [0] * (MAX_DISTANCE + 1)
    for code in range(N_DISTANCE_CODES):
        base, extra = DISTANCE_CODES[code]
        span = 1 << extra
        for off in range(span):
            D = base + off
            if D > MAX_DISTANCE:
                break
            D_to_code[D] = code
            D_to_extra[D] = off
    return D_to_code, D_to_extra


_L_TO_CODE, _L_TO_EXTRA = _build_length_lookup()
_D_TO_CODE, _D_TO_EXTRA = _build_distance_lookup()


def length_to_code(length: int) -> Tuple[int, int, int]:
    """Return (code, extra_value, extra_bit_count)."""
    if length < 3 or length > MAX_LENGTH:
        raise ValueError(f"length {length} out of range 3..{MAX_LENGTH}")
    code = _L_TO_CODE[length]
    return code, _L_TO_EXTRA[length], LENGTH_CODES[code][1]


def code_to_length(code: int, extra_value: int) -> int:
    base, _ = LENGTH_CODES[code]
    return base + extra_value


def distance_to_code(distance: int) -> Tuple[int, int, int]:
    """Return (code, extra_value, extra_bit_count)."""
    if distance < 1 or distance > MAX_DISTANCE:
        raise ValueError(f"distance {distance} out of range 1..{MAX_DISTANCE}")
    code = _D_TO_CODE[distance]
    return code, _D_TO_EXTRA[distance], DISTANCE_CODES[code][1]


def code_to_distance(code: int, extra_value: int) -> int:
    base, _ = DISTANCE_CODES[code]
    return base + extra_value


def length_extra_bits(code: int) -> int:
    return LENGTH_CODES[code][1]


def distance_extra_bits(code: int) -> int:
    return DISTANCE_CODES[code][1]
