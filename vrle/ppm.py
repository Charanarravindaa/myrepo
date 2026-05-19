"""PPM (Prediction by Partial Matching) order-N byte-stream compressor.

The big lever for text compression: model each byte's probability
*conditioned on the last N bytes*, not in isolation. After ``"th"`` the
probability of ``"e"`` is ~25 %, vs ~10 % unconditional — order-0 wastes
that bit budget. PPM gives it back.

Variant: PPM-C-style escape (escape count = number of distinct symbols
seen in the context). Each context starts empty; on the first update it
is seeded with the new symbol (count 1) and an escape symbol (count 1).
Subsequent updates increment the symbol's count and, for new symbols,
the escape count too.

Order-0 is pre-seeded with all 256 byte symbols at count 1 so it never
needs to escape — there's no "order -1" uniform fallback to write.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .bitpack import leb128_decode, leb128_encode
from .rangecoder import (
    MAX_TOTAL,
    ArithmeticDecoder,
    ArithmeticEncoder,
    FenwickFreq,
)

ALPHABET = 256
ESC = ALPHABET  # escape symbol id
TABLE_SIZE = ALPHABET + 1


class PPMContext:
    """Adaptive frequency table for a single context (256 syms + escape)."""

    __slots__ = ("freq",)

    def __init__(self) -> None:
        self.freq = FenwickFreq(TABLE_SIZE)

    @property
    def total(self) -> int:
        return self.freq.total

    def count(self, sym: int) -> int:
        return self.freq.count_at(sym)

    def cumul(self, sym: int) -> int:
        return self.freq.prefix_sum(sym)

    def update(self, sym: int) -> None:
        # First time we see this symbol here? Bump escape count too.
        if self.freq.count_at(sym) == 0:
            self.freq.increment(ESC, 1)
        self.freq.increment(sym, 1)
        if self.freq.total >= MAX_TOTAL:
            self.freq.halve()


def _seed_order0() -> PPMContext:
    """Order-0 context: uniform over the byte alphabet, no escape needed."""
    ctx = PPMContext()
    for s in range(ALPHABET):
        ctx.freq.increment(s, 1)
    return ctx


def _grab_context(history: bytearray, k: int) -> bytes:
    """Return the last `k` bytes of history as a bytes key."""
    if k == 0:
        return b""
    if k >= len(history):
        return bytes(history)
    return bytes(history[-k:])


def _ctx_counts_list(ctx: PPMContext) -> List[int]:
    """Materialise the count vector (length TABLE_SIZE) for fast scanning.

    Fenwick is O(log α) per query but iterating all α symbols repeatedly
    under exclusion is faster as a flat list.
    """
    return [ctx.freq.count_at(i) for i in range(TABLE_SIZE)]


def encode_ppm_stream(data: bytes, order: int = 4) -> bytes:
    """Compress `data` with order-N PPM-C, with full exclusion, on top of
    adaptive arithmetic coding.

    Exclusion: once a higher-order context has escaped, every symbol that
    was *in* that context is removed from the model at every lower order
    for the rest of this single-byte encoding step. The distribution at
    lower orders tightens — typical ~5–10 % gain on natural text.

    Layout: leb128(n) [|| leb128(order) leb128(n_bits) leb128(payload_len)
    payload].
    """
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)
    out += leb128_encode(order)

    contexts: List[Dict[bytes, PPMContext]] = [dict() for _ in range(order + 1)]
    contexts[0][b""] = _seed_order0()

    enc = ArithmeticEncoder()
    history = bytearray()
    excluded = bytearray(TABLE_SIZE)  # 1 = excluded; reused, reset per byte

    for sym in data:
        # Reset exclusion mask for this byte.
        for i in range(TABLE_SIZE):
            excluded[i] = 0

        for k in range(min(order, len(history)), -1, -1):
            ctx = contexts[k].get(_grab_context(history, k))
            if ctx is None or ctx.total == 0:
                continue
            counts = _ctx_counts_list(ctx)
            # Effective totals/cumuls under exclusion.
            eff_total = 0
            for i in range(TABLE_SIZE):
                if not excluded[i]:
                    eff_total += counts[i]
            if eff_total <= 0:
                continue

            if not excluded[sym] and counts[sym] > 0:
                # Encode sym at this order using effective cumulative.
                lo = 0
                for i in range(sym):
                    if not excluded[i]:
                        lo += counts[i]
                enc.encode_freq(lo, lo + counts[sym], eff_total)
                break

            # Encode escape (if it has a count) and exclude every symbol
            # seen in this context from lower orders.
            esc_count = counts[ESC]
            if esc_count > 0:
                lo = 0
                for i in range(ESC):
                    if not excluded[i]:
                        lo += counts[i]
                enc.encode_freq(lo, lo + esc_count, eff_total)
            # Exclude every present symbol (not the escape itself).
            for i in range(ESC):
                if counts[i] > 0:
                    excluded[i] = 1

        # Update every context (no exclusion in updates).
        for k in range(min(order, len(history)), -1, -1):
            key = _grab_context(history, k)
            ctx = contexts[k].get(key)
            if ctx is None:
                ctx = PPMContext()
                contexts[k][key] = ctx
            ctx.update(sym)

        history.append(sym)
        if len(history) > order:
            del history[: len(history) - order]

    payload, n_bits = enc.finish()
    out += leb128_encode(n_bits)
    out += leb128_encode(len(payload))
    out += payload
    return bytes(out)


def decode_ppm_stream(buf: bytes, pos: int = 0) -> Tuple[bytes, int]:
    n, pos = leb128_decode(buf, pos)
    if n == 0:
        return b"", pos
    order, pos = leb128_decode(buf, pos)
    n_bits, pos = leb128_decode(buf, pos)
    payload_len, pos = leb128_decode(buf, pos)
    payload = buf[pos : pos + payload_len]
    pos += payload_len

    contexts: List[Dict[bytes, PPMContext]] = [dict() for _ in range(order + 1)]
    contexts[0][b""] = _seed_order0()

    dec = ArithmeticDecoder(payload, n_bits)
    out = bytearray()
    history = bytearray()
    excluded = bytearray(TABLE_SIZE)

    for _ in range(n):
        for i in range(TABLE_SIZE):
            excluded[i] = 0
        sym = -1
        for k in range(min(order, len(history)), -1, -1):
            ctx = contexts[k].get(_grab_context(history, k))
            if ctx is None or ctx.total == 0:
                continue
            counts = _ctx_counts_list(ctx)
            # Effective cumulative array
            eff_cumul = [0] * (TABLE_SIZE + 1)
            for i in range(TABLE_SIZE):
                eff_cumul[i + 1] = eff_cumul[i] + (0 if excluded[i] else counts[i])
            eff_total = eff_cumul[-1]
            if eff_total <= 0:
                continue
            scaled = dec.scale_value(eff_total)
            # Binary search for the symbol s with eff_cumul[s] <= scaled < eff_cumul[s+1].
            lo, hi = 0, TABLE_SIZE - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if eff_cumul[mid] <= scaled:
                    lo = mid
                else:
                    hi = mid - 1
            s = lo
            lo_c, hi_c = eff_cumul[s], eff_cumul[s + 1]
            dec.consume_freq(lo_c, hi_c, eff_total)
            if s == ESC:
                # Continue down; exclude all present symbols.
                for i in range(ESC):
                    if counts[i] > 0:
                        excluded[i] = 1
                continue
            sym = s
            break

        if sym < 0:
            raise RuntimeError("PPM decode fell off the chain")

        out.append(sym)

        for k in range(min(order, len(history)), -1, -1):
            key = _grab_context(history, k)
            ctx = contexts[k].get(key)
            if ctx is None:
                ctx = PPMContext()
                contexts[k][key] = ctx
            ctx.update(sym)

        history.append(sym)
        if len(history) > order:
            del history[: len(history) - order]

    return bytes(out), pos
