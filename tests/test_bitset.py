from geomds import GeometricBitset


def test_add_coalesces_contiguous():
    b = GeometricBitset()
    for i in range(100):
        b.add(i)
    assert b.runs() == 1
    assert len(b) == 100


def test_add_merges_adjacent_runs():
    b = GeometricBitset()
    for i in range(10):
        b.add(i)
    for i in range(11, 20):
        b.add(i)
    assert b.runs() == 2
    b.add(10)
    assert b.runs() == 1
    assert len(b) == 20


def test_membership():
    b = GeometricBitset([1, 2, 3, 7, 8])
    assert 1 in b
    assert 3 in b
    assert 4 not in b
    assert 7 in b
    assert 9 not in b


def test_discard_splits_run():
    b = GeometricBitset(range(10))
    assert b.runs() == 1
    b.discard(5)
    assert b.runs() == 2
    assert 5 not in b
    assert 4 in b and 6 in b
    assert len(b) == 9


def test_discard_endpoints():
    b = GeometricBitset(range(10))
    b.discard(0)
    b.discard(9)
    assert b.runs() == 1
    assert len(b) == 8


def test_union_intersection():
    a = GeometricBitset(range(0, 100))
    b = GeometricBitset(range(50, 150))
    u = a | b
    assert len(u) == 150
    assert u.runs() == 1
    i = a & b
    assert len(i) == 50
    assert i.runs() == 1


def test_disjoint_union():
    a = GeometricBitset([1, 2, 3])
    b = GeometricBitset([10, 11, 12])
    u = a | b
    assert u.runs() == 2
    assert len(u) == 6


def test_iter_order():
    b = GeometricBitset([5, 3, 1, 2, 4])
    assert list(b) == [1, 2, 3, 4, 5]
    assert b.runs() == 1
