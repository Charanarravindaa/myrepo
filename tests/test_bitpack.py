import random

import pytest

from vrle.bitpack import (
    BitReader,
    BitWriter,
    gamma_bits,
    leb128_decode,
    leb128_encode,
    pack_runs,
    unpack_runs,
)
from vrle.core import Run


@pytest.mark.parametrize("n", [0, 1, 2, 127, 128, 255, 256, 16383, 16384, 1_000_000, 2**32 - 1])
def test_leb128_roundtrip(n):
    buf = leb128_encode(n)
    decoded, pos = leb128_decode(buf, 0)
    assert decoded == n
    assert pos == len(buf)


def test_leb128_rejects_negative():
    with pytest.raises(ValueError):
        leb128_encode(-1)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 1023, 1024, 65535, 999_999])
def test_gamma_roundtrip(n):
    w = BitWriter()
    w.write_gamma(n)
    assert len(w) == gamma_bits(n)
    r = BitReader(w.to_bytes(), len(w))
    assert r.read_gamma() == n


def test_gamma_rejects_non_positive():
    w = BitWriter()
    with pytest.raises(ValueError):
        w.write_gamma(0)


def test_bit_writer_reader_mixed():
    w = BitWriter()
    w.write_bits(0b1011, 4)
    w.write_gamma(5)
    w.write_bits(0xFF, 8)
    r = BitReader(w.to_bytes(), len(w))
    assert r.read_bits(4) == 0b1011
    assert r.read_gamma() == 5
    assert r.read_bits(8) == 0xFF


@pytest.mark.parametrize("encoding", ["leb128", "gamma", "raw32"])
def test_pack_unpack_runs_roundtrip(encoding):
    rng = random.Random(42)
    runs = [Run(rng.randrange(256), rng.randint(1, 5000)) for _ in range(50)]
    buf, bit_len = pack_runs(runs, token_width=8, freq_encoding=encoding)
    decoded = unpack_runs(buf, bit_len, n_runs=len(runs), token_width=8, freq_encoding=encoding)
    assert decoded == runs


def test_pack_rejects_token_too_wide():
    with pytest.raises(ValueError):
        pack_runs([Run(256, 1)], token_width=8, freq_encoding="leb128")


def test_gamma_bit_widths_are_correct():
    # Sanity: width(n)=floor(log2(n))+1, encoding uses 2*width-1 bits
    assert gamma_bits(1) == 1
    assert gamma_bits(2) == 3
    assert gamma_bits(3) == 3
    assert gamma_bits(4) == 5
    assert gamma_bits(7) == 5
    assert gamma_bits(8) == 7
