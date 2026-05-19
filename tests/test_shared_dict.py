from pathlib import Path

import pytest

from vrle.shared_dict import SharedDict


def test_basic_construction():
    d = SharedDict([b"the", b" ", b"cat"], version=2)
    assert len(d) == 3
    assert d.version == 2
    assert d.get_id(b"the") == 0
    assert d.get_id(b" ") == 1
    assert d.get_id(b"cat") == 2
    assert d.get_id(b"unknown") is None
    assert d.get_token(0) == b"the"
    assert b"the" in d
    assert b"missing" not in d


def test_duplicate_rejected():
    with pytest.raises(ValueError):
        SharedDict([b"a", b"b", b"a"])


def test_roundtrip():
    d = SharedDict([b"hello", b" ", b"world", b"!"], version=7)
    blob = d.serialize()
    d2 = SharedDict.deserialize(blob)
    assert d2.tokens == d.tokens
    assert d2.version == d.version


def test_empty_dict():
    d = SharedDict([], version=0)
    assert len(d) == 0
    blob = d.serialize()
    d2 = SharedDict.deserialize(blob)
    assert d2.tokens == []


def test_save_load(tmp_path):
    d = SharedDict([b"alpha", b"beta", b"gamma"], version=42)
    path = tmp_path / "d.bin"
    d.save(path)
    d2 = SharedDict.load(path)
    assert d2.tokens == d.tokens
    assert d2.version == d.version


def test_real_bundled_dict_loads():
    p = Path(__file__).resolve().parent.parent / "examples" / "data" / "base_english.dict"
    if not p.is_file():
        pytest.skip("base_english.dict not present")
    d = SharedDict.load(p)
    assert len(d) > 100
    assert d.version >= 1
    # Common English words should be in there.
    for w in (b"the", b" ", b"of", b"and", b"to"):
        assert d.get_id(w) is not None, f"common token {w!r} missing from base dict"
