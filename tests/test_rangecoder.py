import random

import pytest

from vrle.rangecoder import (
    FrequencyTable,
    build_frequency_table,
    decode_stream,
    encode_stream,
)


def test_table_construction_and_lookup():
    t = FrequencyTable([3, 2, 5])  # cumul = [0, 3, 5, 10], total = 10
    assert t.total == 10
    assert t.cumul_range(0) == (0, 3)
    assert t.cumul_range(1) == (3, 5)
    assert t.cumul_range(2) == (5, 10)
    assert t.find(0) == 0
    assert t.find(2) == 0
    assert t.find(3) == 1
    assert t.find(4) == 1
    assert t.find(5) == 2
    assert t.find(9) == 2


def test_table_total_zero_rejected():
    with pytest.raises(ValueError):
        FrequencyTable([0, 0, 0])


@pytest.mark.parametrize(
    "symbols, alphabet",
    [
        ([0], 1),
        ([0, 0, 0], 1),
        ([0, 1, 2, 3], 4),
        (list(range(10)) * 5, 10),
    ],
)
def test_roundtrip_small(symbols, alphabet):
    blob = encode_stream(symbols, alphabet)
    decoded, _ = decode_stream(blob)
    assert decoded == symbols


def test_roundtrip_random_bytes():
    rng = random.Random(99)
    symbols = [rng.randrange(256) for _ in range(10_000)]
    blob = encode_stream(symbols, 256)
    decoded, _ = decode_stream(blob)
    assert decoded == symbols


def test_roundtrip_skewed_distribution():
    rng = random.Random(100)
    # 90% zeros, 10% other bytes
    symbols = [
        0 if rng.random() < 0.9 else rng.randrange(1, 256) for _ in range(20_000)
    ]
    blob = encode_stream(symbols, 256)
    decoded, _ = decode_stream(blob)
    assert decoded == symbols
    # On a 90% zero stream the entropy is < 0.5 bits/symbol, so the blob
    # should be well under raw 8 bits/symbol = 20_000 bytes.
    assert len(blob) < 4_000


def test_empty_stream():
    blob = encode_stream([], 256)
    # Self-delimiting: empty stream costs exactly one leb128(0) byte.
    assert blob == b"\x00"
    decoded, pos = decode_stream(blob)
    assert decoded == []
    assert pos == 1


def test_build_frequency_table_scales_when_oversized():
    rng = random.Random(101)
    symbols = [rng.randrange(256) for _ in range(100_000)]
    table = build_frequency_table(symbols, 256)
    assert table.total <= (1 << 14)
    # Every symbol that appeared should have count >= 1.
    for sym in set(symbols):
        assert table.counts[sym] >= 1
