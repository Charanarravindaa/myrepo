import pytest

from geomds import GeometricStack


def test_push_pop_lifo():
    s = GeometricStack[int]()
    for x in [1, 2, 2, 3, 3, 3]:
        s.push(x)
    assert len(s) == 6
    assert list(s) == [1, 2, 2, 3, 3, 3]
    assert [s.pop() for _ in range(6)] == [3, 3, 3, 2, 2, 1]
    assert len(s) == 0


def test_coalescing():
    s = GeometricStack[str]()
    for _ in range(1000):
        s.push("a")
    assert len(s) == 1000
    assert s.runs() == 1
    assert s.compression() == 1000.0


def test_push_many():
    s = GeometricStack[str]()
    s.push_many("a", 10**6)
    s.push_many("b", 10**6)
    assert len(s) == 2 * 10**6
    assert s.runs() == 2
    assert s.pop() == "b"
    assert s.runs() == 2


def test_empty_errors():
    s = GeometricStack[int]()
    with pytest.raises(IndexError):
        s.pop()
    with pytest.raises(IndexError):
        s.peek()


def test_alternating():
    s = GeometricStack[int]()
    for x in [1, 2, 1, 2, 1]:
        s.push(x)
    assert s.runs() == 5
    assert len(s) == 5
