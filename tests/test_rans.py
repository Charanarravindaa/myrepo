import random

import pytest

from vrle.rans import (
    PROB_TOTAL,
    normalize_counts,
    rans_decode,
    rans_encode,
)


def test_normalize_sums_to_total():
    raw = [10, 20, 5, 0, 7]
    n = normalize_counts(raw)
    assert sum(n) == PROB_TOTAL
    # Zero stays zero; non-zero stays >= 1.
    assert n[3] == 0
    assert all(n[i] >= 1 for i in (0, 1, 2, 4))


def test_normalize_extreme_skew():
    raw = [1_000_000] + [1] * 50
    n = normalize_counts(raw)
    assert sum(n) == PROB_TOTAL
    assert all(n[i] >= 1 for i in range(1, 51))


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        normalize_counts([0, 0, 0])


def test_normalize_rejects_negative():
    with pytest.raises(ValueError):
        normalize_counts([1, -1, 2])


@pytest.mark.parametrize(
    "symbols, raw",
    [
        ([0], [1]),
        ([0, 0, 0], [3]),
        ([0, 1, 0, 1], [2, 2]),
        ([0, 1, 2, 0, 1, 0], [3, 2, 1]),
        ([i % 4 for i in range(200)], [50, 50, 50, 50]),
    ],
)
def test_roundtrip_known(symbols, raw):
    blob = rans_encode(symbols, raw)
    decoded = rans_decode(blob, raw, len(symbols))
    assert decoded == symbols


def test_roundtrip_random_byte_alphabet():
    rng = random.Random(505)
    raw = [rng.randint(1, 100) for _ in range(256)]
    # Generate symbols proportional to the freqs.
    population = list(range(256))
    symbols = rng.choices(population, weights=raw, k=10_000)
    blob = rans_encode(symbols, raw)
    decoded = rans_decode(blob, raw, len(symbols))
    assert decoded == symbols


def test_roundtrip_skewed():
    # 90% one symbol, 10% others
    raw = [9000] + [100] * 10
    rng = random.Random(606)
    population = list(range(len(raw)))
    symbols = rng.choices(population, weights=raw, k=5_000)
    blob = rans_encode(symbols, raw)
    decoded = rans_decode(blob, raw, len(symbols))
    assert decoded == symbols
    # Very skewed -> very compressible. Should beat 1 byte per symbol.
    assert len(blob) < 1500


def test_empty_input():
    assert rans_encode([], [1, 2, 3]) == b""
    assert rans_decode(b"", [1, 2, 3], 0) == []


def test_zero_freq_symbol_rejected():
    with pytest.raises(ValueError):
        # Symbol 2 is in the stream but has zero raw frequency.
        rans_encode([0, 2], [5, 5, 0])


def test_large_symbol_count():
    """A long stream over a small alphabet exercises renormalisation."""
    rng = random.Random(707)
    raw = [50, 30, 15, 5]
    population = [0, 1, 2, 3]
    symbols = rng.choices(population, weights=raw, k=50_000)
    blob = rans_encode(symbols, raw)
    decoded = rans_decode(blob, raw, len(symbols))
    assert decoded == symbols
