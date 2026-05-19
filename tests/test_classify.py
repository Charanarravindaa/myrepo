import random
from pathlib import Path

import pytest

from vrle import classify as cls


def _rand(n, seed=1):
    r = random.Random(seed)
    return bytes(r.randrange(256) for _ in range(n))


def _runs(n, alphabet=4, seed=2):
    r = random.Random(seed)
    out = bytearray()
    while len(out) < n:
        out.extend([r.randrange(alphabet)] * r.randint(40, 200))
    return bytes(out[:n])


def test_empty():
    assert cls.classify(b"") == cls.UNKNOWN


def test_random_bytes():
    assert cls.classify(_rand(20_000)) == cls.RANDOM


def test_long_runs():
    assert cls.classify(_runs(20_000)) == cls.REDUNDANT


def test_english_prose():
    p = Path("examples/data/sample.txt")
    if not p.is_file():
        pytest.skip("sample.txt missing")
    assert cls.classify(p.read_bytes()) == cls.ENGLISH


def test_english_long_text():
    p = Path("examples/data/long_text.txt")
    if not p.is_file():
        pytest.skip("long_text.txt missing")
    assert cls.classify(p.read_bytes()) == cls.ENGLISH


def test_log_stream():
    rng = random.Random(4)
    lines = []
    for _ in range(500):
        lines.append(
            f"{rng.choice(['GET', 'POST'])} {rng.choice(['/api/v1/users', '/static/app.css'])} HTTP/1.1 {rng.choice([200, 404])}\n"
        )
    assert cls.classify("".join(lines).encode()) == cls.LOGS


def test_source_code():
    # Use our own pipelines.py — substantial Python source.
    p = Path("vrle/pipelines.py")
    assert cls.classify(p.read_bytes()) == cls.CODE


def test_features_return_dict():
    f = cls.features(b"hello world")
    assert "entropy" in f
    assert "word_frac" in f
    assert 0 <= f["word_frac"] <= 1
