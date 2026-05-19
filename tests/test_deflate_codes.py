import pytest

from vrle.deflate_codes import (
    MAX_DISTANCE,
    MAX_LENGTH,
    N_DISTANCE_CODES,
    N_LENGTH_CODES,
    code_to_distance,
    code_to_length,
    distance_extra_bits,
    distance_to_code,
    length_extra_bits,
    length_to_code,
)


def test_all_lengths_roundtrip():
    for L in range(3, MAX_LENGTH + 1):
        code, extra, bits = length_to_code(L)
        assert 0 <= code < N_LENGTH_CODES
        assert bits == length_extra_bits(code)
        assert 0 <= extra < (1 << bits) if bits else extra == 0
        assert code_to_length(code, extra) == L


def test_all_distances_roundtrip():
    for D in range(1, MAX_DISTANCE + 1):
        code, extra, bits = distance_to_code(D)
        assert 0 <= code < N_DISTANCE_CODES
        assert bits == distance_extra_bits(code)
        assert 0 <= extra < (1 << bits) if bits else extra == 0
        assert code_to_distance(code, extra) == D


def test_length_258_is_special_code():
    code, extra, bits = length_to_code(258)
    assert code == 28
    assert extra == 0
    assert bits == 0


def test_length_257_uses_code_27():
    code, extra, bits = length_to_code(257)
    assert code == 27
    assert bits == 5


def test_length_out_of_range_raises():
    with pytest.raises(ValueError):
        length_to_code(2)
    with pytest.raises(ValueError):
        length_to_code(259)


def test_distance_out_of_range_raises():
    with pytest.raises(ValueError):
        distance_to_code(0)
    with pytest.raises(ValueError):
        distance_to_code(MAX_DISTANCE + 1)


def test_short_distances_have_no_extra_bits():
    for D in range(1, 5):
        _, _, bits = distance_to_code(D)
        assert bits == 0
