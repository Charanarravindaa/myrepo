from pathlib import Path

import pytest

from vrle.random_access import (
    RandomReader,
    compress_random,
    decompress_random,
)
from vrle.wordtok import tokenize


SAMPLES = [
    b"",
    b"a",
    b"hello world",
    b"the quick brown fox jumps over the lazy dog. " * 50,
    b"GET /index.html HTTP/1.1 200 OK\n" * 100,
]


@pytest.mark.parametrize("data", SAMPLES)
@pytest.mark.parametrize("block_size", [128, 512, 2048])
def test_roundtrip(data, block_size):
    blob = compress_random(data, block_size=block_size)
    assert decompress_random(blob) == data


def test_roundtrip_long_text():
    data = Path("examples/data/long_text.txt").read_bytes()
    blob = compress_random(data)
    assert decompress_random(blob) == data


def test_reader_introspection():
    data = b"the quick brown fox jumps over the lazy dog. " * 20
    blob = compress_random(data, block_size=64)
    r = RandomReader(blob)
    assert r.total_bytes == len(data)
    assert r.block_size == 64
    assert r.n_blocks == (r.n_tokens + 63) // 64
    assert r.read_all() == data


def test_read_word_matches_tokenize():
    data = Path("examples/data/long_text.txt").read_bytes()
    expected = tokenize(data)
    blob = compress_random(data)
    r = RandomReader(blob)
    assert r.n_tokens == len(expected)
    # Spot-check at various positions: start, mid, end, boundaries.
    for k in (0, 1, 5, 100, 500, r.block_size - 1, r.block_size, r.n_tokens - 1):
        if k < r.n_tokens:
            assert r.read_word(k) == expected[k], f"mismatch at word {k}"


def test_read_word_index_errors():
    data = b"hello world"
    blob = compress_random(data)
    r = RandomReader(blob)
    with pytest.raises(IndexError):
        r.read_word(-1)
    with pytest.raises(IndexError):
        r.read_word(r.n_tokens)


@pytest.mark.parametrize(
    "start, end",
    [
        (0, 10),
        (0, 100),
        (100, 200),
        (1000, 1100),
        (5000, 6000),
        (0, 53241),  # whole file
        (50000, 53241),  # tail
        (26000, 27000),  # mid-block
    ],
)
def test_read_byte_range(start, end):
    data = Path("examples/data/long_text.txt").read_bytes()
    blob = compress_random(data, block_size=2048)
    r = RandomReader(blob)
    assert r.read_byte_range(start, end) == data[start:end]


def test_read_byte_range_empty():
    data = b"hello world"
    blob = compress_random(data)
    r = RandomReader(blob)
    assert r.read_byte_range(0, 0) == b""
    assert r.read_byte_range(5, 5) == b""


def test_read_byte_range_invalid():
    data = b"hello world"
    blob = compress_random(data)
    r = RandomReader(blob)
    with pytest.raises(IndexError):
        r.read_byte_range(-1, 5)
    with pytest.raises(IndexError):
        r.read_byte_range(0, len(data) + 1)
    with pytest.raises(IndexError):
        r.read_byte_range(5, 2)


def test_block_split_works():
    """A many-block file must round-trip byte-exact."""
    data = (
        b"This is line number %d in a moderately repetitive document. " % i
        for i in range(200)
    )
    blob_in = b"".join(data)
    blob = compress_random(blob_in, block_size=64)
    r = RandomReader(blob)
    assert r.n_blocks >= 5  # sanity
    assert r.read_all() == blob_in
    # Random reads across the boundaries.
    for k in range(0, r.n_tokens, 7):
        assert isinstance(r.read_word(k), bytes)


def test_empty_blob_rejected():
    with pytest.raises(ValueError):
        RandomReader(b"")


def test_empty_input_decompresses_to_empty():
    blob = compress_random(b"")
    assert decompress_random(blob) == b""
