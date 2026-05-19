import random

import pytest

from vrle.ppm import decode_ppm_stream, encode_ppm_stream


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"aa",
        b"abcabcabc",
        b"the quick brown fox jumps over the lazy dog",
        b"x" * 200,
        bytes(range(256)),
    ],
)
@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_roundtrip_known(data, order):
    blob = encode_ppm_stream(data, order=order)
    decoded, _ = decode_ppm_stream(blob)
    assert decoded == data


def test_roundtrip_random_bytes():
    rng = random.Random(303)
    data = bytes(rng.randrange(256) for _ in range(2_000))
    blob = encode_ppm_stream(data, order=4)
    decoded, _ = decode_ppm_stream(blob)
    assert decoded == data


def test_roundtrip_repetitive_text():
    data = b"the quick brown fox jumps over the lazy dog. " * 50
    blob = encode_ppm_stream(data, order=4)
    decoded, _ = decode_ppm_stream(blob)
    assert decoded == data
    # Repetitive text — should compress hard.
    assert len(blob) < len(data) // 5


def test_higher_order_compresses_text_better_than_order0():
    data = b"the quick brown fox jumps over the lazy dog. " * 30
    o0 = encode_ppm_stream(data, order=0)
    o4 = encode_ppm_stream(data, order=4)
    assert len(o4) < len(o0)


def test_empty_input():
    blob = encode_ppm_stream(b"", order=4)
    decoded, _ = decode_ppm_stream(blob)
    assert decoded == b""
