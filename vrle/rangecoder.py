"""Arithmetic coder (32-bit, bit-oriented) with semi-adaptive byte model.

This is the entropy step of every v2 pipeline. The encoder narrows an
integer interval `[low, high]` with each symbol; renormalisation emits
bits as the top of the interval settles. Bit-oriented form is slow but
simple to reason about and easy to round-trip reliably.

The frequency table is semi-adaptive: the encoder counts symbols in a
single pass, scales the histogram to total <= MAX_TOTAL, writes the
scaled histogram into a small header, then encodes against it. The
decoder reads the header, rebuilds the model, and inverts.
"""

from __future__ import annotations

from typing import List, Tuple

from .bitpack import leb128_decode, leb128_encode

PRECISION = 32
FULL = 1 << PRECISION
HALF = FULL >> 1
QUARTER = FULL >> 2
THREE_QUARTERS = HALF + QUARTER
MASK = FULL - 1

# Frequency totals must stay below QUARTER so the interval narrowing
# preserves precision. We use 2**14 = 16384 as a safe target.
MAX_TOTAL = 1 << 14


# ---------------------------------------------------------------------------
# Frequency table
# ---------------------------------------------------------------------------


class FrequencyTable:
    """Cumulative frequency table over an alphabet of size `n`.

    `cumul[i]` = sum of counts of all symbols with id < i. `cumul[n]` is
    the total. Counts for absent symbols are zero — they cannot be
    encoded against this table.
    """

    __slots__ = ("counts", "cumul", "total")

    def __init__(self, counts: List[int]) -> None:
        if not counts:
            raise ValueError("FrequencyTable needs a non-empty alphabet")
        self.counts = list(counts)
        self.cumul = [0] * (len(counts) + 1)
        for i, c in enumerate(self.counts):
            if c < 0:
                raise ValueError("counts must be non-negative")
            self.cumul[i + 1] = self.cumul[i] + c
        self.total = self.cumul[-1]
        if self.total == 0:
            raise ValueError("FrequencyTable total cannot be zero")
        if self.total >= QUARTER:
            raise ValueError(
                f"FrequencyTable total {self.total} exceeds limit {QUARTER}"
            )

    def cumul_range(self, sym: int) -> Tuple[int, int]:
        return self.cumul[sym], self.cumul[sym + 1]

    def find(self, scaled_value: int) -> int:
        # Binary search for the symbol with cumul[s] <= scaled_value < cumul[s+1]
        lo, hi = 0, len(self.counts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cumul[mid] <= scaled_value:
                lo = mid
            else:
                hi = mid - 1
        return lo


def build_frequency_table(symbols: List[int], alphabet_size: int) -> FrequencyTable:
    counts = [0] * alphabet_size
    for s in symbols:
        counts[s] += 1
    # Scale down so that total <= MAX_TOTAL while preserving non-zero counts.
    total = sum(counts)
    if total == 0:
        raise ValueError("Cannot build frequency table from empty stream")
    if total > MAX_TOTAL:
        # Proportional scaling with a floor of 1 for any non-zero symbol.
        scale = MAX_TOTAL / total
        scaled = [max(1 if c else 0, int(c * scale)) for c in counts]
        # Adjust so the sum lands exactly at MAX_TOTAL (or close): trim/grow
        # the largest counts to absorb the difference.
        diff = MAX_TOTAL - sum(scaled)
        if diff != 0:
            order = sorted(
                range(alphabet_size), key=lambda i: -scaled[i]
            )
            j = 0
            step = 1 if diff > 0 else -1
            remaining = abs(diff)
            while remaining > 0:
                idx = order[j % len(order)]
                if step < 0 and scaled[idx] <= 1 and counts[idx] > 0:
                    j += 1
                    if j > 10 * len(order):
                        break  # give up — close enough
                    continue
                scaled[idx] += step
                remaining -= 1
                j += 1
        counts = scaled
    return FrequencyTable(counts)


# ---------------------------------------------------------------------------
# Header encoding for the frequency table
# ---------------------------------------------------------------------------


def encode_table(table: FrequencyTable) -> bytes:
    """Encode the alphabet size + sparse non-zero counts as LEB128 fields."""
    out = bytearray()
    out += leb128_encode(len(table.counts))
    non_zero = [(i, c) for i, c in enumerate(table.counts) if c]
    out += leb128_encode(len(non_zero))
    for sym, count in non_zero:
        out += leb128_encode(sym)
        out += leb128_encode(count)
    return bytes(out)


def decode_table(buf: bytes, pos: int) -> Tuple[FrequencyTable, int]:
    alphabet_size, pos = leb128_decode(buf, pos)
    n_nonzero, pos = leb128_decode(buf, pos)
    counts = [0] * alphabet_size
    for _ in range(n_nonzero):
        sym, pos = leb128_decode(buf, pos)
        count, pos = leb128_decode(buf, pos)
        counts[sym] = count
    return FrequencyTable(counts), pos


# ---------------------------------------------------------------------------
# Arithmetic encoder / decoder
# ---------------------------------------------------------------------------


class ArithmeticEncoder:
    __slots__ = ("low", "high", "pending", "_bits")

    def __init__(self) -> None:
        self.low = 0
        self.high = MASK
        self.pending = 0
        self._bits: List[int] = []

    def _emit(self, bit: int) -> None:
        self._bits.append(bit)
        for _ in range(self.pending):
            self._bits.append(1 - bit)
        self.pending = 0

    def encode(self, table: FrequencyTable, sym: int) -> None:
        lo, hi = table.cumul_range(sym)
        width = self.high - self.low + 1
        self.high = self.low + width * hi // table.total - 1
        self.low = self.low + width * lo // table.total
        # Renormalise
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK

    def finish(self) -> Tuple[bytes, int]:
        self.pending += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        n_bits = len(self._bits)
        # pad to byte boundary
        while len(self._bits) % 8:
            self._bits.append(0)
        out = bytearray(len(self._bits) // 8)
        for i, b in enumerate(self._bits):
            if b:
                out[i >> 3] |= 1 << (7 - (i & 7))
        return bytes(out), n_bits


class ArithmeticDecoder:
    __slots__ = ("data", "n_bits", "bit_pos", "value", "low", "high")

    def __init__(self, data: bytes, n_bits: int) -> None:
        self.data = data
        self.n_bits = n_bits
        self.bit_pos = 0
        self.low = 0
        self.high = MASK
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self.bit_pos >= self.n_bits:
            return 0
        byte = self.data[self.bit_pos >> 3]
        bit = (byte >> (7 - (self.bit_pos & 7))) & 1
        self.bit_pos += 1
        return bit

    def decode(self, table: FrequencyTable) -> int:
        width = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * table.total - 1) // width
        sym = table.find(scaled)
        lo, hi = table.cumul_range(sym)
        self.high = self.low + width * hi // table.total - 1
        self.low = self.low + width * lo // table.total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK
            self.value = ((self.value << 1) | self._read_bit()) & MASK
        return sym


# ---------------------------------------------------------------------------
# Stream-level helpers
# ---------------------------------------------------------------------------


def encode_stream(symbols: List[int], alphabet_size: int) -> bytes:
    """Encode `symbols` to a self-contained byte string.

    Layout: leb128(n_symbols) [|| encode_table(model) || leb128(n_bits) ||
    leb128(payload_len) || payload].  The leading symbol-count is always
    present so an empty stream still consumes one byte and decoders can
    chain calls via the returned position.
    """
    out = bytearray()
    out += leb128_encode(len(symbols))
    if not symbols:
        return bytes(out)
    table = build_frequency_table(symbols, alphabet_size)
    enc = ArithmeticEncoder()
    for s in symbols:
        enc.encode(table, s)
    payload, n_bits = enc.finish()
    out += encode_table(table)
    out += leb128_encode(n_bits)
    out += leb128_encode(len(payload))
    out += payload
    return bytes(out)


def decode_stream(buf: bytes, pos: int = 0) -> Tuple[List[int], int]:
    n_symbols, pos = leb128_decode(buf, pos)
    if n_symbols == 0:
        return [], pos
    table, pos = decode_table(buf, pos)
    n_bits, pos = leb128_decode(buf, pos)
    payload_len, pos = leb128_decode(buf, pos)
    payload = buf[pos : pos + payload_len]
    pos += payload_len
    dec = ArithmeticDecoder(payload, n_bits)
    out = [dec.decode(table) for _ in range(n_symbols)]
    return out, pos
