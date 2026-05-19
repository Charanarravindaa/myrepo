import pytest

from vrle.vocab import Vocabulary


def test_from_tokens_basic():
    toks = [b"the", b" ", b"cat", b" ", b"the", b" ", b"dog"]
    v = Vocabulary.from_tokens(toks)
    # "the" appears twice, " " appears 3 times — so " " should be the
    # most frequent and get ID 0.
    assert v.tokens[0] == b" "
    assert v.freqs[0] == 3
    assert v.total == len(toks)


def test_id_roundtrip():
    toks = [b"a", b" ", b"b", b" ", b"a"]
    v = Vocabulary.from_tokens(toks)
    ids = v.tokens_to_ids(toks)
    assert v.ids_to_tokens(ids) == toks


def test_serialize_deserialize():
    toks = [b"hello", b" ", b"world", b"!", b" ", b"hello"]
    v = Vocabulary.from_tokens(toks)
    blob = v.serialize()
    v2, end = Vocabulary.deserialize(blob)
    assert end == len(blob)
    assert v.tokens == v2.tokens
    assert v.freqs == v2.freqs
    assert v.total == v2.total


def test_serialize_with_offset():
    toks = [b"x", b"y", b"x"]
    v = Vocabulary.from_tokens(toks)
    prefix = b"--ignore-this--"
    buf = prefix + v.serialize() + b"trailing"
    v2, end = Vocabulary.deserialize(buf, len(prefix))
    assert v.tokens == v2.tokens
    assert v.freqs == v2.freqs
    assert buf[end:] == b"trailing"


def test_empty_input():
    v = Vocabulary.from_tokens([])
    assert len(v) == 0
    assert v.freqs == []
    blob = v.serialize()
    v2, end = Vocabulary.deserialize(blob)
    assert end == len(blob)
    assert len(v2) == 0


def test_freqs_are_descending():
    toks = [b"a"] * 10 + [b"b"] * 5 + [b"c"] * 1
    v = Vocabulary.from_tokens(toks)
    assert v.freqs == sorted(v.freqs, reverse=True)
