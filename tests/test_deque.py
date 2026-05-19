import pytest

from geomds import GeometricDeque


def test_both_ends():
    d = GeometricDeque[int]()
    d.push_back(1)
    d.push_back(2)
    d.push_front(0)
    d.push_front(0)
    assert list(d) == [0, 0, 1, 2]
    assert d.pop_front() == 0
    assert d.pop_back() == 2


def test_coalesce_both_ends():
    d = GeometricDeque[str]()
    d.push_back("x", 100)
    d.push_back("x", 100)
    d.push_front("x", 100)
    assert d.runs() == 1
    assert len(d) == 300


def test_empty_errors():
    d = GeometricDeque[int]()
    with pytest.raises(IndexError):
        d.pop_front()
    with pytest.raises(IndexError):
        d.pop_back()
