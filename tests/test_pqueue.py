import pytest

from geomds import GeometricPriorityQueue


def test_streaming_top_k():
    pq = GeometricPriorityQueue[str]()
    for tok in "aaabbcccccd":
        pq.push(tok)
    assert pq.peek_max() == ("c", 5)
    tok, mag = pq.pop_max()
    assert tok == "c" and mag == 5
    assert pq.peek_max() == ("a", 3)


def test_promotion_on_push():
    pq = GeometricPriorityQueue[str]()
    pq.push("a", 5)
    pq.push("b", 3)
    assert pq.peek_max() == ("a", 5)
    pq.push("b", 10)
    assert pq.peek_max() == ("b", 13)


def test_pop_empty():
    pq = GeometricPriorityQueue[int]()
    with pytest.raises(IndexError):
        pq.pop_max()


def test_distinct_and_len():
    pq = GeometricPriorityQueue[str]()
    pq.push("a", 4)
    pq.push("b", 6)
    assert len(pq) == 10
    assert pq.distinct() == 2
