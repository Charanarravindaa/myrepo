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


def encode_ppm_stream(data: bytes, order: int = 4) -> bytes:
    """Compress `data` with order-N PPM-C and adaptive arithmetic coding.

    Layout: leb128(n) [|| leb128(order) leb128(n_bits) leb128(payload_len)
    payload].
    """
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)
    out += leb128_encode(order)

    # contexts[k] : dict[ bytes-of-length-k -> PPMContext ]
    contexts: List[Dict[bytes, PPMContext]] = [dict() for _ in range(order + 1)]
    contexts[0][b""] = _seed_order0()

    enc = ArithmeticEncoder()
    history = bytearray()

    for sym in data:
        # Try contexts from longest to shortest. Order-0 is guaranteed to
        # succeed (uniform seed), so the loop always terminates with the
        # symbol encoded.
        for k in range(min(order, len(history)), -1, -1):
            ctx = contexts[k].get(_grab_context(history, k))
            if ctx is None or ctx.total == 0:
                continue
            sym_count = ctx.count(sym)
            if sym_count > 0:
                lo = ctx.cumul(sym)
                enc.encode_freq(lo, lo + sym_count, ctx.total)
                break
            # Escape and fall to a shorter context. Every context past its
            # first update has escape count >= 1.
            esc_count = ctx.count(ESC)
            if esc_count > 0:
                lo = ctx.cumul(ESC)
                enc.encode_freq(lo, lo + esc_count, ctx.total)
            # If esc_count == 0 the context has no symbols at all
            # (shouldn't happen after first update) — silently skip.

        # Update every context in the chain.
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

    for _ in range(n):
        sym = -1
        for k in range(min(order, len(history)), -1, -1):
            ctx = contexts[k].get(_grab_context(history, k))
            if ctx is None or ctx.total == 0:
                continue
            scaled = dec.scale_value(ctx.total)
            s = ctx.freq.find_sym(scaled)
            if s == ESC:
                # Consume the escape range and continue down.
                lo = ctx.cumul(ESC)
                dec.consume_freq(lo, lo + ctx.count(ESC), ctx.total)
                continue
            # Real symbol — consume and stop.
            lo = ctx.cumul(s)
            dec.consume_freq(lo, lo + ctx.count(s), ctx.total)
            sym = s
            break

        if sym < 0:
            # Should not happen: order-0 always succeeds.
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
