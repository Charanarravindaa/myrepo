# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython acceleration for the PPM hot path.

Profile of ``vrle/ppm.py:_encode_loop`` shows ~95 % of compression time
spent in the alphabet-sized inner loops — building effective cumulative
arrays under exclusion, summing effective totals, walking the alphabet
to mark excluded symbols.

These helpers move each of those loops into compiled C. The rest of
``_encode_loop`` (context lookup, history management, PPM state
machine) stays Python; it's branchy and operates on dict / tuple
objects that don't accelerate cleanly.

All helpers take ``unsigned char[::1]`` memoryviews so they accept
both ``bytes`` (read-only views) and ``bytearray`` (writable views)
without copying.
"""

cimport cython
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from cpython.long cimport PyLong_AsLong


@cython.boundscheck(False)
@cython.wraparound(False)
def eff_total(list counts, const unsigned char[::1] excluded):
    cdef Py_ssize_t n = PyList_GET_SIZE(counts)
    cdef Py_ssize_t i
    cdef long total = 0
    for i in range(n):
        if excluded[i] == 0:
            total += PyLong_AsLong(<object>PyList_GET_ITEM(counts, i))
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
def eff_cumul_lo(list counts, const unsigned char[::1] excluded, int sym):
    cdef Py_ssize_t i
    cdef long lo = 0
    for i in range(sym):
        if excluded[i] == 0:
            lo += PyLong_AsLong(<object>PyList_GET_ITEM(counts, i))
    return lo


@cython.boundscheck(False)
@cython.wraparound(False)
def build_eff_cumul(list counts, const unsigned char[::1] excluded):
    cdef Py_ssize_t n = PyList_GET_SIZE(counts)
    cdef Py_ssize_t i
    cdef long running = 0
    out = [0] * (n + 1)
    for i in range(n):
        if excluded[i] == 0:
            running += PyLong_AsLong(<object>PyList_GET_ITEM(counts, i))
        out[i + 1] = running
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def reset_excluded(unsigned char[::1] excluded):
    cdef Py_ssize_t n = excluded.shape[0]
    cdef Py_ssize_t i
    for i in range(n):
        excluded[i] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
def mark_excluded(unsigned char[::1] excluded, list counts, int up_to):
    cdef Py_ssize_t i
    cdef long c
    for i in range(up_to):
        if excluded[i] == 0:
            c = PyLong_AsLong(<object>PyList_GET_ITEM(counts, i))
            if c > 0:
                excluded[i] = 1


@cython.boundscheck(False)
@cython.wraparound(False)
def find_sym_in_cumul(list eff_cumul, long scaled, int table_size):
    """Binary search: largest sym with eff_cumul[sym] <= scaled."""
    cdef int lo = 0
    cdef int hi = table_size - 1
    cdef int mid
    cdef long mid_val
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        mid_val = PyLong_AsLong(<object>PyList_GET_ITEM(eff_cumul, mid))
        if mid_val <= scaled:
            lo = mid
        else:
            hi = mid - 1
    return lo
