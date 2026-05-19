import pytest

from geomds import GeometricQueue


def test_fifo():
    q = GeometricQueue[int]()
    for x in [1, 2, 2, 3]:
        q.enqueue(x)
    assert [q.dequeue() for _ in range(4)] == [1, 2, 2, 3]


def test_coalescing_tail():
    q = GeometricQueue[str]()
    for _ in range(1000):
        q.enqueue("a")
    assert q.runs() == 1


def test_enqueue_many():
    q = GeometricQueue[str]()
    q.enqueue_many("x", 10**6)
    assert len(q) == 10**6
    assert q.runs() == 1
    assert q.dequeue() == "x"
    assert len(q) == 10**6 - 1
    assert q.runs() == 1


def test_empty_errors():
    q = GeometricQueue[int]()
    with pytest.raises(IndexError):
        q.dequeue()
