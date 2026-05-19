import random

import pytest

from vrle.wordtok import detokenize, tokenize


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"hello",
        b"hello world",
        b"hello, world!",
        b"  spaces   matter  ",
        b"The quick brown fox jumps over the lazy dog.",
        b"abc123_def 456",
        b"\n\n\t  \r\n",
        bytes(range(256)),
    ],
)
def test_roundtrip(data):
    assert detokenize(tokenize(data)) == data


def test_empty():
    assert tokenize(b"") == []
    assert detokenize([]) == b""


def test_word_run():
    assert tokenize(b"hello") == [b"hello"]


def test_word_then_punct():
    assert tokenize(b"hi!") == [b"hi", b"!"]


def test_alternating_classes():
    assert tokenize(b"a b c") == [b"a", b" ", b"b", b" ", b"c"]


def test_utf8_pass_through():
    # "café" in UTF-8 — the multibyte é (0xC3 0xA9) should stay inside
    # one word token because high bytes are word bytes.
    data = "café".encode()
    toks = tokenize(data)
    assert detokenize(toks) == data
    assert len(toks) == 1


def test_punct_run():
    assert tokenize(b"...!!!") == [b"...!!!"]


def test_random_bytes_roundtrip():
    rng = random.Random(404)
    data = bytes(rng.randrange(256) for _ in range(5_000))
    assert detokenize(tokenize(data)) == data
