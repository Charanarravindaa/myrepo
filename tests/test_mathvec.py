import math

import pytest

from vrle.mathvec import MathRLE
from vrle.sequence import SequenceRLE


def test_from_iterable_counts_tokens():
    v = MathRLE.from_iterable("aaabbbccccba")
    assert v["a"] == 4
    assert v["b"] == 4
    assert v["c"] == 4
    assert v["missing"] == 0
    assert len(v) == 3


def test_zero_vector():
    z = MathRLE()
    assert len(z) == 0
    assert z["anything"] == 0


def test_add_is_union_with_sum():
    a = MathRLE.from_iterable("aaabb")
    b = MathRLE.from_iterable("bccc")
    s = a + b
    assert s.magnitudes == {"a": 3, "b": 3, "c": 3}


def test_add_zero_identity():
    v = MathRLE.from_iterable("xxyz")
    assert (v + MathRLE()) == v
    assert (MathRLE() + v) == v


def test_scalar_zero_yields_zero():
    v = MathRLE.from_iterable("xxyz")
    assert (v * 0) == MathRLE()


def test_scalar_multiply_scales_all():
    v = MathRLE.from_iterable("aabbb")
    scaled = v * 3
    assert scaled.magnitudes == {"a": 6, "b": 9}


def test_rmul_works():
    v = MathRLE.from_iterable("aab")
    assert (2 * v).magnitudes == {"a": 4, "b": 2}


def test_dot_product():
    a = MathRLE({"x": 2, "y": 3})
    b = MathRLE({"x": 4, "y": 5, "z": 10})
    # 2*4 + 3*5 + 0*10 = 23
    assert a.dot(b) == 23
    assert a.dot(b) == b.dot(a)


def test_cosine_self_is_one():
    v = MathRLE.from_iterable("aaabbc")
    assert math.isclose(v.cosine_similarity(v), 1.0)


def test_cosine_orthogonal_is_zero():
    a = MathRLE({"x": 5})
    b = MathRLE({"y": 7})
    assert a.cosine_similarity(b) == 0.0


def test_cosine_with_zero_vector():
    v = MathRLE.from_iterable("ab")
    assert v.cosine_similarity(MathRLE()) == 0.0


def test_from_sequence_collapses_positional_runs():
    s = SequenceRLE.from_iterable("aaabbbaaa")
    v = MathRLE.from_sequence(s)
    # a appears in two runs (3 + 3); they should collapse to 6
    assert v.magnitudes == {"a": 6, "b": 3}


def test_negative_magnitude_rejected():
    with pytest.raises(ValueError):
        MathRLE({"x": -1})


def test_negative_scalar_rejected():
    with pytest.raises(ValueError):
        MathRLE({"x": 1}) * -2
