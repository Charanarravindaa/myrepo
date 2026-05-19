import random

import pytest

from vrle.bwt import bwt_decode, bwt_encode


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"banana",
        b"abracadabra",
        b"the quick brown fox jumps over the lazy dog",
        b"aaaaaaaaaa",
        b"abcabcabcabc",
        bytes(range(256)),
    ],
)
def test_roundtrip(data):
    transformed, primary = bwt_encode(data)
    assert bwt_decode(transformed, primary) == data
    if data:
        assert len(transformed) == len(data)


def test_random_bytes_roundtrip():
    rng = random.Random(13)
    for size in (1, 10, 100, 1_000):
        data = bytes(rng.randrange(256) for _ in range(size))
        transformed, primary = bwt_encode(data)
        assert bwt_decode(transformed, primary) == data


def test_long_run_clusters_to_last_column():
    data = b"a" * 50
    transformed, _ = bwt_encode(data)
    # Every rotation of "aaaa...a" puts 'a' in the last column.
    assert transformed == data


def test_banana_known_result():
    # Classic textbook example: BWT("banana") with implicit cyclic rotation,
    # last column = "nnbaaa", primary index = 3.
    transformed, primary = bwt_encode(b"banana")
    assert transformed == b"nnbaaa"
    assert primary == 3
