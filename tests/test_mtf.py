import random

import pytest

from vrle.mtf import mtf_decode, mtf_encode


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"aaaa",
        b"abcabc",
        b"the quick brown fox jumps over the lazy dog",
        bytes(range(256)),
    ],
)
def test_roundtrip(data):
    assert mtf_decode(mtf_encode(data)) == data


def test_random_roundtrip():
    rng = random.Random(7)
    data = bytes(rng.randrange(256) for _ in range(5_000))
    assert mtf_decode(mtf_encode(data)) == data


def test_long_run_collapses_to_zeros():
    data = b"x" * 100
    encoded = mtf_encode(data)
    # First byte = position of 'x' in fresh table (120). All subsequent = 0.
    assert encoded[0] == ord("x")
    assert all(b == 0 for b in encoded[1:])


def test_clustered_input_is_mostly_small():
    """After BWT, clusters are common — MTF should produce mostly small ints."""
    data = b"aaaa" + b"bbbb" + b"cccc" + b"dddd"
    encoded = mtf_encode(data)
    # First of each cluster is the symbol's position; the rest are 0.
    assert encoded.count(0) >= 12
