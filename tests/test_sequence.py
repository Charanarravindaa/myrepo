import random

import pytest

from vrle.core import Run
from vrle.sequence import SequenceRLE


@pytest.mark.parametrize(
    "seq",
    [
        b"",
        b"a",
        b"aaaa",
        b"abcd",
        b"aaabbbcccaaa",
        b"the quick brown fox jumps over the lazy dog",
    ],
)
def test_bytes_roundtrip(seq):
    rle = SequenceRLE.from_iterable(seq)
    assert bytes(rle.to_iterable()) == seq


def test_string_roundtrip():
    s = "aaabbbcccdddddeeeee"
    rle = SequenceRLE.from_iterable(s)
    assert "".join(rle.to_iterable()) == s


def test_arbitrary_token_roundtrip():
    tokens = ["GET", "GET", "GET", "POST", "POST", "DELETE", "DELETE", "DELETE", "DELETE"]
    rle = SequenceRLE.from_iterable(tokens)
    assert rle.to_iterable() == tokens
    assert rle.runs == [Run("GET", 3), Run("POST", 2), Run("DELETE", 4)]


def test_iter_streams_in_order():
    seq = "aaabbc"
    rle = SequenceRLE.from_iterable(seq)
    assert list(rle) == list(seq)


def test_len_is_original_length():
    rle = SequenceRLE.from_iterable("aaabbc")
    assert len(rle) == 6


def test_concat_fuses_boundary_run():
    a = SequenceRLE.from_iterable("aaab")
    b = SequenceRLE.from_iterable("bbcc")
    merged = a.concat(b)
    assert merged.runs == [Run("a", 3), Run("b", 3), Run("c", 2)]


def test_concat_no_fuse():
    a = SequenceRLE.from_iterable("aaab")
    b = SequenceRLE.from_iterable("ccd")
    merged = a.concat(b)
    assert merged.runs == [Run("a", 3), Run("b", 1), Run("c", 2), Run("d", 1)]


def test_concat_with_empty():
    a = SequenceRLE.from_iterable("")
    b = SequenceRLE.from_iterable("xy")
    assert a.concat(b).to_iterable() == ["x", "y"]
    assert b.concat(a).to_iterable() == ["x", "y"]


def test_long_run_property():
    rng = random.Random(0)
    parts = []
    for _ in range(100):
        tok = rng.randrange(4)
        parts.extend([tok] * rng.randint(1, 50))
    rle = SequenceRLE.from_iterable(parts)
    assert rle.to_iterable() == parts


def test_bit_size_strictly_smaller_than_raw_on_runs():
    data = list(b"a" * 1000 + b"b" * 1000 + b"c" * 1000)
    rle = SequenceRLE.from_iterable(data)
    raw_bits = len(data) * 8
    assert rle.bit_size(8, "leb128") < raw_bits
    assert rle.bit_size(8, "gamma") < raw_bits


def test_bit_size_raw32_baseline():
    rle = SequenceRLE.from_iterable("aabbcc")
    # 3 runs * (8 + 32) = 120 bits
    assert rle.bit_size(8, "raw32") == 120
