import pytest

from geomds import GeometricMultiset


def test_count_and_iter():
    m = GeometricMultiset("aaabbc")
    assert m["a"] == 3
    assert m["b"] == 2
    assert m["c"] == 1
    assert m["z"] == 0
    assert len(m) == 6
    assert m.distinct() == 3


def test_union_intersection():
    a = GeometricMultiset("aaabb")
    b = GeometricMultiset("abbcc")
    assert (a | b)._counts == {"a": 3, "b": 2, "c": 2}
    assert (a & b)._counts == {"a": 1, "b": 2}


def test_sum_difference():
    a = GeometricMultiset("aaabb")
    b = GeometricMultiset("abbcc")
    assert (a + b)._counts == {"a": 4, "b": 4, "c": 2}
    assert (a - b)._counts == {"a": 2}


def test_dot_cosine():
    a = GeometricMultiset("aaab")
    b = GeometricMultiset("aabbb")
    assert a.dot(b) == 3 * 2 + 1 * 3
    same = GeometricMultiset("xy")
    assert same.cosine(same) == pytest.approx(1.0)


def test_cosine_empty():
    a = GeometricMultiset()
    b = GeometricMultiset("x")
    assert a.cosine(b) == 0.0
