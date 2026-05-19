import random

import pytest

from vrle.pipelines import ALL_PIPELINES, bwt_mtf_rle_rc, mtf_rle_rc, rle_rc


SAMPLES = [
    b"",
    b"a",
    b"aaaaaaaaaa",
    b"the quick brown fox jumps over the lazy dog",
    b"abracadabra" * 50,
    b"GET /index.html HTTP/1.1 200 OK\n" * 100,
]


@pytest.mark.parametrize("pipeline", ALL_PIPELINES, ids=lambda p: p.name)
@pytest.mark.parametrize("data", SAMPLES, ids=[f"len{len(s)}" for s in SAMPLES])
def test_roundtrip_known_samples(pipeline, data):
    blob = pipeline.compress(data)
    assert pipeline.decompress(blob) == data


@pytest.mark.parametrize("pipeline", ALL_PIPELINES, ids=lambda p: p.name)
def test_roundtrip_random_bytes(pipeline):
    rng = random.Random(42)
    data = bytes(rng.randrange(256) for _ in range(2_000))
    blob = pipeline.compress(data)
    assert pipeline.decompress(blob) == data


@pytest.mark.parametrize("pipeline", ALL_PIPELINES, ids=lambda p: p.name)
def test_roundtrip_long_runs(pipeline):
    data = b"".join(bytes([i % 7]) * (50 + i) for i in range(40))
    blob = pipeline.compress(data)
    assert pipeline.decompress(blob) == data


def test_bwt_pipeline_crushes_high_redundancy():
    data = b"abcdefgh" * 5_000  # 40 000 bytes of pure repetition
    blob = bwt_mtf_rle_rc.compress(data)
    assert bwt_mtf_rle_rc.decompress(blob) == data
    assert len(blob) < len(data) // 100  # under 1% of raw


def test_rle_rc_beats_naive_on_runs():
    data = b"a" * 1_000 + b"b" * 1_000 + b"c" * 1_000
    blob = rle_rc.compress(data)
    assert rle_rc.decompress(blob) == data
    # Trivially: 3 000 bytes -> some tens of bytes.
    assert len(blob) < 200


def test_pipelines_dont_blow_up_on_random():
    """Random data is incompressible; we just need round-trip + bounded overhead."""
    rng = random.Random(7)
    data = bytes(rng.randrange(256) for _ in range(5_000))
    for p in ALL_PIPELINES:
        blob = p.compress(data)
        assert p.decompress(blob) == data
        # Overhead vs raw should be under 2x even on random data.
        assert len(blob) < 2 * len(data)
