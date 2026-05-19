"""PPM (Prediction by Partial Matching) — order-N context model.

Generic over alphabet size. Used in two places:

* ``encode_ppm_stream`` / ``decode_ppm_stream`` — byte-level (alphabet
  256), the same API the v4 pipelines rely on.
* ``encode_ppm_seq`` / ``decode_ppm_seq`` — general integer sequences
  with arbitrary alphabet size, used by ``vector_rle`` in v6 to model
  word-bigram structure on the positional index.

Variant: PPM-C-style escape with full exclusion. Each context starts
empty; on the first update it is seeded with the new symbol (count 1)
and an escape symbol (count 1). On every escape, all symbols seen in
the higher-order context are excluded from the model at every lower
order for the remainder of that single-byte encoding step.

Order-0 is pre-seeded with all alphabet symbols at count 1 so it never
needs to escape — there is no order -1 uniform fallback to encode.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .bitpack import leb128_decode, leb128_encode
from .rangecoder import (
    MAX_TOTAL,
    ArithmeticDecoder,
    ArithmeticEncoder,
)


class PPMContext:
    """Direct count vector for a single context. Length = alphabet+1
    (the final slot is the escape symbol).
    """

    __slots__ = ("counts", "total")

    def __init__(self, alphabet_size: int) -> None:
        self.counts = [0] * (alphabet_size + 1)
        self.total = 0

    def update(self, sym: int) -> None:
        esc = len(self.counts) - 1
        if self.counts[sym] == 0:
            self.counts[esc] += 1
            self.total += 1
        self.counts[sym] += 1
        self.total += 1
        if self.total >= MAX_TOTAL:
            new_total = 0
            for i, c in enumerate(self.counts):
                if c > 0:
                    nc = max(1, c // 2)
                    self.counts[i] = nc
                    new_total += nc
            self.total = new_total


def _seed_order0(alphabet_size: int) -> PPMContext:
    """Order-0 context: every alphabet symbol at count 1, never escapes."""
    ctx = PPMContext(alphabet_size)
    ctx.counts = [1] * alphabet_size + [0]
    ctx.total = alphabet_size
    return ctx


def _ctx_key(history: List[int], k: int) -> Tuple[int, ...]:
    if k == 0:
        return ()
    if k >= len(history):
        return tuple(history)
    return tuple(history[-k:])


def _encode_loop(
    symbols: List[int],
    alphabet_size: int,
    order: int,
    enc: ArithmeticEncoder,
) -> None:
    esc = alphabet_size
    table_size = alphabet_size + 1

    contexts: List[Dict[Tuple[int, ...], PPMContext]] = [
        dict() for _ in range(order + 1)
    ]
    contexts[0][()] = _seed_order0(alphabet_size)

    history: List[int] = []
    excluded = bytearray(table_size)

    for sym in symbols:
        for i in range(table_size):
            excluded[i] = 0

        for k in range(min(order, len(history)), -1, -1):
            ctx = contexts[k].get(_ctx_key(history, k))
            if ctx is None or ctx.total == 0:
                continue
            counts = ctx.counts

            # Effective total under exclusion.
            eff_total = 0
            for i in range(table_size):
                if not excluded[i]:
                    eff_total += counts[i]
            if eff_total <= 0:
                continue

            if not excluded[sym] and counts[sym] > 0:
                lo = 0
                for i in range(sym):
                    if not excluded[i]:
                        lo += counts[i]
                enc.encode_freq(lo, lo + counts[sym], eff_total)
                break

            esc_count = counts[esc]
            if esc_count > 0:
                lo = 0
                for i in range(esc):
                    if not excluded[i]:
                        lo += counts[i]
                enc.encode_freq(lo, lo + esc_count, eff_total)

            for i in range(esc):
                if counts[i] > 0:
                    excluded[i] = 1

        # Update every context (no exclusion in updates).
        for k in range(min(order, len(history)), -1, -1):
            key = _ctx_key(history, k)
            ctx = contexts[k].get(key)
            if ctx is None:
                ctx = PPMContext(alphabet_size)
                contexts[k][key] = ctx
            ctx.update(sym)

        history.append(sym)
        if len(history) > order:
            del history[: len(history) - order]


def _decode_loop(
    alphabet_size: int,
    order: int,
    n_symbols: int,
    dec: ArithmeticDecoder,
) -> List[int]:
    esc = alphabet_size
    table_size = alphabet_size + 1

    contexts: List[Dict[Tuple[int, ...], PPMContext]] = [
        dict() for _ in range(order + 1)
    ]
    contexts[0][()] = _seed_order0(alphabet_size)

    out: List[int] = []
    history: List[int] = []
    excluded = bytearray(table_size)

    for _ in range(n_symbols):
        for i in range(table_size):
            excluded[i] = 0
        sym = -1

        for k in range(min(order, len(history)), -1, -1):
            ctx = contexts[k].get(_ctx_key(history, k))
            if ctx is None or ctx.total == 0:
                continue
            counts = ctx.counts

            # Build effective cumulative array.
            eff_cumul = [0] * (table_size + 1)
            for i in range(table_size):
                eff_cumul[i + 1] = eff_cumul[i] + (
                    0 if excluded[i] else counts[i]
                )
            eff_total = eff_cumul[-1]
            if eff_total <= 0:
                continue

            scaled = dec.scale_value(eff_total)
            lo_i, hi_i = 0, table_size - 1
            while lo_i < hi_i:
                mid = (lo_i + hi_i + 1) // 2
                if eff_cumul[mid] <= scaled:
                    lo_i = mid
                else:
                    hi_i = mid - 1
            s = lo_i

            dec.consume_freq(eff_cumul[s], eff_cumul[s + 1], eff_total)
            if s == esc:
                for i in range(esc):
                    if counts[i] > 0:
                        excluded[i] = 1
                continue
            sym = s
            break

        if sym < 0:
            raise RuntimeError("PPM decode fell off the chain")

        out.append(sym)
        for k in range(min(order, len(history)), -1, -1):
            key = _ctx_key(history, k)
            ctx = contexts[k].get(key)
            if ctx is None:
                ctx = PPMContext(alphabet_size)
                contexts[k][key] = ctx
            ctx.update(sym)

        history.append(sym)
        if len(history) > order:
            del history[: len(history) - order]

    return out


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def encode_ppm_seq(symbols: List[int], alphabet_size: int, order: int = 4) -> bytes:
    """Encode an arbitrary integer sequence with PPM-C with exclusion.

    Layout: leb128(n) [|| leb128(alphabet_size) leb128(order)
    leb128(n_bits) leb128(payload_len) payload].
    """
    out = bytearray()
    out += leb128_encode(len(symbols))
    if not symbols:
        return bytes(out)
    out += leb128_encode(alphabet_size)
    out += leb128_encode(order)

    enc = ArithmeticEncoder()
    _encode_loop(symbols, alphabet_size, order, enc)
    payload, n_bits = enc.finish()
    out += leb128_encode(n_bits)
    out += leb128_encode(len(payload))
    out += payload
    return bytes(out)


def decode_ppm_seq(buf: bytes, pos: int = 0) -> Tuple[List[int], int]:
    n, pos = leb128_decode(buf, pos)
    if n == 0:
        return [], pos
    alphabet_size, pos = leb128_decode(buf, pos)
    order, pos = leb128_decode(buf, pos)
    n_bits, pos = leb128_decode(buf, pos)
    payload_len, pos = leb128_decode(buf, pos)
    payload = buf[pos : pos + payload_len]
    pos += payload_len

    dec = ArithmeticDecoder(payload, n_bits)
    out = _decode_loop(alphabet_size, order, n, dec)
    return out, pos


def encode_ppm_stream(data: bytes, order: int = 4) -> bytes:
    """Byte-level PPM-C (alphabet=256).

    Layout (unchanged from v4): leb128(n) [|| leb128(order)
    leb128(n_bits) leb128(payload_len) payload].
    """
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)
    out += leb128_encode(order)

    enc = ArithmeticEncoder()
    _encode_loop(list(data), 256, order, enc)
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

    dec = ArithmeticDecoder(payload, n_bits)
    out = _decode_loop(256, order, n, dec)
    return bytes(out), pos
