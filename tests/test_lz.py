import random

import pytest

from vrle.lz import lz77_decode, lz77_encode


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"abcabc",
        b"the quick brown fox jumps over the lazy dog",
        b"abracadabra" * 20,
        b"GET /index.html HTTP/1.1\n" * 50,
        b"x" * 500,
    ],
)
def test_roundtrip(data):
    controls, literals, lengths, distances = lz77_encode(data)
    assert lz77_decode(controls, literals, lengths, distances) == data


def test_random_bytes_roundtrip():
    rng = random.Random(11)
    data = bytes(rng.randrange(256) for _ in range(5_000))
    streams = lz77_encode(data)
    assert lz77_decode(*streams) == data


def test_long_run_is_mostly_matches():
    data = b"x" * 200
    controls, literals, lengths, distances = lz77_encode(data)
    # First byte must be a literal (nothing to match against yet); after
    # that the encoder is free to find a back-reference of distance 1.
    assert controls[0] == 0
    assert controls.count(1) >= 1
    # Reconstruct length sum + literal count = original length.
    assert sum(lengths) + len(literals) == len(data)


def test_overlap_copy_works():
    """A back-reference where length > distance must copy byte-by-byte."""
    # 'a' repeated: after first 3 bytes, a match (distance=1, length=N) refers
    # to overlapping data. The decoder must handle this.
    data = b"a" * 100
    controls, literals, lengths, distances = lz77_encode(data)
    assert lz77_decode(controls, literals, lengths, distances) == data


def test_match_counts_consistent():
    data = b"the quick brown fox " * 100
    controls, literals, lengths, distances = lz77_encode(data)
    assert controls.count(0) == len(literals)
    assert controls.count(1) == len(lengths) == len(distances)
