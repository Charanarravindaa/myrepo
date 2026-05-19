"""Byte-level word tokenizer.

Splits input bytes into a stream of tokens whose concatenation is exactly
the original input (`detokenize(tokenize(x)) == x` for every `x`). A
token is a maximal run of either *word* characters or *non-word*
characters; runs of mixed classes are split at every class change.

A *word* character is `[A-Za-z0-9_]` (ASCII letters, digits, underscore)
or any byte with the high bit set (>=128). The high-bit rule keeps
UTF-8 continuation bytes attached to the word run, so multi-byte UTF-8
characters end up inside a single word token instead of being shattered.

The empty input produces an empty token list. Round-trip is exact.
"""

from __future__ import annotations

from typing import List


def _is_word_byte(b: int) -> bool:
    # ASCII letters
    if 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A:
        return True
    # Digits
    if 0x30 <= b <= 0x39:
        return True
    # Underscore
    if b == 0x5F:
        return True
    # High-byte (UTF-8 continuation / leading bytes for multibyte chars)
    if b >= 0x80:
        return True
    return False


MAX_TOKEN_LEN = 255  # so length fits in a single byte


def tokenize(data: bytes) -> List[bytes]:
    if not data:
        return []
    tokens: List[bytes] = []
    start = 0
    cur_class = _is_word_byte(data[0])

    def _emit(s: int, e: int) -> None:
        # Split very long maximal runs at the byte-length cap so every
        # token's length fits in a single byte (used by vector_rle).
        while e - s > MAX_TOKEN_LEN:
            tokens.append(bytes(data[s : s + MAX_TOKEN_LEN]))
            s += MAX_TOKEN_LEN
        if e > s:
            tokens.append(bytes(data[s:e]))

    for i in range(1, len(data)):
        cls = _is_word_byte(data[i])
        if cls != cur_class:
            _emit(start, i)
            start = i
            cur_class = cls
    _emit(start, len(data))
    return tokens


def detokenize(tokens: List[bytes]) -> bytes:
    return b"".join(tokens)
