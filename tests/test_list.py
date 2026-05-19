from geomds import GeometricList


def test_append_coalesces():
    g = GeometricList[str]()
    for c in "aaabbbbcc":
        g.append(c)
    assert g.runs() == 3
    assert list(g) == list("aaabbbbcc")


def test_prepend_coalesces():
    g = GeometricList[str]()
    g.prepend("a", 5)
    g.prepend("a", 5)
    assert g.runs() == 1
    assert len(g) == 10


def test_splice_merges_at_join():
    a = GeometricList[str]()
    a.append("x", 3)
    a.append("y", 2)
    b = GeometricList[str]()
    b.append("y", 4)
    b.append("z", 1)
    a.splice(b)
    assert list(a) == list("xxxyyyyyyz")
    assert a.runs() == 3
    assert len(b) == 0


def test_splice_no_merge():
    a = GeometricList[str]()
    a.append("x", 3)
    b = GeometricList[str]()
    b.append("y", 3)
    a.splice(b)
    assert a.runs() == 2
    assert list(a) == list("xxxyyy")


def test_splice_into_empty():
    a = GeometricList[str]()
    b = GeometricList[str]()
    b.append("z", 5)
    a.splice(b)
    assert len(a) == 5
    assert a.runs() == 1
    assert len(b) == 0
