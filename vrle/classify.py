"""Content classifier for ``vector_auto``.

Extracts six cheap features from a ~4 KB sample of the input (three
chunks: start, middle, end — so structural signals survive front-matter
like docstrings or copyright headers) and returns a category tag that
``vector_auto`` uses to pick the best underlying compression strategy
plus shared dictionary.

Categories:
  RANDOM     — incompressible, fall back to byte-level PPM
  REDUNDANT  — long runs of repeated bytes
  ENGLISH    — natural-language prose
  LOGS       — line-oriented log streams
  CODE       — source code (many operator characters and newlines)
  UNKNOWN    — binary / mixed; byte-level PPM
"""

from __future__ import annotations

import math
from collections import Counter

from .wordtok import _is_word_byte

RANDOM = "random"
REDUNDANT = "redundant"
ENGLISH = "english"
LOGS = "logs"
CODE = "code"
ARITH = "arith"  # text-y but no word structure (e.g. single-letter tokens)
UNKNOWN = "unknown"

SAMPLE_BYTES = 4096


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _word_byte_frac(data: bytes) -> float:
    return sum(1 for b in data if _is_word_byte(b)) / max(1, len(data))


def _newline_density(data: bytes) -> float:
    return data.count(b"\n") / max(1, len(data))


def _ascii_printable_frac(data: bytes) -> float:
    return sum(
        1 for b in data if (0x20 <= b <= 0x7E) or b in (0x09, 0x0A, 0x0D)
    ) / max(1, len(data))


def _code_density(data: bytes) -> float:
    """Fraction of operator-ish characters common in source code but
    relatively rare in prose: ``()[]{}=:;,./``.
    """
    n = max(1, len(data))
    return sum(data.count(c) for c in b"()[]{}=:;,./") / n


# Substrings that almost always indicate source code (multi-language).
_CODE_KEYWORDS = (
    b"def ", b"class ", b"import ", b"return ", b"function ",
    b" if (", b"} else", b"else if", b"#include", b"#define",
    b"public ", b"private ", b"static ", b"void ", b" const ",
    b" let ", b" var ", b"->",
)

# Substrings that almost always indicate log lines.
_LOG_PATTERNS = (
    b"HTTP/", b" GET ", b" POST ", b" PUT ", b" DELETE ",
    b" 200 ", b" 201 ", b" 301 ", b" 304 ", b" 400 ", b" 404 ", b" 500 ",
    b"INFO ", b"ERROR ", b"WARN ", b"DEBUG ", b"TRACE ",
    b"[INFO]", b"[ERROR]", b"[WARN]",
    b" - - [",
)


def _count_patterns(data: bytes, patterns) -> int:
    return sum(data.count(p) for p in patterns)


def _avg_token_len(data: bytes) -> float:
    """Average length of the word tokens in `data`. Single-letter
    streams report ~1.0; real prose ~3-5; logs ~3-4."""
    from .wordtok import tokenize
    tokens = tokenize(data)
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def _avg_run_length(data: bytes) -> float:
    if not data:
        return 0.0
    runs = 1
    last = data[0]
    for b in data[1:]:
        if b != last:
            runs += 1
            last = b
    return len(data) / runs


def _multi_sample(data: bytes, sample: int) -> bytes:
    """Sample from start, middle, and end so heavy front-matter
    (docstrings, license headers) doesn't dominate the signal."""
    n = len(data)
    if n <= sample:
        return bytes(data)
    chunk = sample // 3
    parts = [
        data[0:chunk],
        data[n // 2 : n // 2 + chunk],
        data[max(0, n - chunk) : n],
    ]
    return b"".join(parts)


def classify(data: bytes, sample: int = SAMPLE_BYTES) -> str:
    if not data:
        return UNKNOWN
    s = _multi_sample(data, sample)

    entropy = _entropy(s)
    word_frac = _word_byte_frac(s)
    newline_dens = _newline_density(s)
    ascii_frac = _ascii_printable_frac(s)
    code_dens = _code_density(s)
    run_len = _avg_run_length(s)

    # Highly redundant first — long runs override all other signals.
    if run_len >= 5.0:
        return REDUNDANT

    # Random / incompressible: near-uniform byte distribution, mostly non-ASCII.
    if entropy > 7.5 and ascii_frac < 0.85:
        return RANDOM

    # Non-text binary: not enough printable to use word tokenisation.
    if ascii_frac < 0.7:
        return UNKNOWN

    # Text-y but with no word structure (avg token length < ~2 bytes
    # means tokens are single letters / single punct chars). Word-level
    # compression has no leverage; pure adaptive arithmetic wins.
    avg_tok = _avg_token_len(s)
    if avg_tok < 1.7:
        return ARITH

    # In text territory. Keyword matches break the tie between code
    # and logs (both have newlines + operator density).
    log_hits = _count_patterns(s, _LOG_PATTERNS)
    code_hits = _count_patterns(s, _CODE_KEYWORDS)

    if log_hits >= 3 and log_hits >= code_hits:
        return LOGS
    if code_hits >= 3 and code_hits > log_hits:
        return CODE

    # Default English text if word-heavy.
    if word_frac > 0.6:
        return ENGLISH

    return UNKNOWN


def features(data: bytes, sample: int = SAMPLE_BYTES) -> dict:
    """Return the raw feature vector (for debugging / introspection)."""
    s = _multi_sample(data, sample)
    return {
        "entropy": _entropy(s),
        "word_frac": _word_byte_frac(s),
        "newline_dens": _newline_density(s),
        "ascii_frac": _ascii_printable_frac(s),
        "code_dens": _code_density(s),
        "run_len": _avg_run_length(s),
        "sample_len": len(s),
    }
