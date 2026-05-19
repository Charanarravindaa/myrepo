"""rANS (range Asymmetric Numeral Systems) entropy coder.

Implementation of static rANS following Duda (2014) and Fabian Giesen's
``rans_byte`` reference. 32-bit state, 8-bit byte renormalisation,
14-bit probability resolution. Static model — the caller supplies a
frequency vector that the symbols' positions are coded against. The
frequency vector is *not* serialized by this module (the caller stores
it; in our v5 pipeline it is the vocabulary's magnitude vector).

The encoder consumes symbols in reverse order and emits bytes that the
decoder reads forward. Output layout:

    [4 bytes: final state, little-endian]
    [body bytes: ordered for the decoder]

Decoder reads the 4-byte final state, then walks the body forward.

Compression bound: within a fraction of a bit per symbol of the
information-theoretic optimum for the given (normalised) model.
"""

from __future__ import annotations

from typing import List

PROB_BITS = 14
PROB_TOTAL = 1 << PROB_BITS  # 16384

# 8-bit byte renormalisation: state lives in [L, 2^32) after renorm.
RANS_BYTE_L = 1 << 23
RANS_BYTE_HI = 1 << 32

_RENORM_FACTOR = (RANS_BYTE_L >> PROB_BITS) << 8  # = 2^17


def normalize_counts(raw: List[int], total: int = PROB_TOTAL) -> List[int]:
    """Scale raw counts so they sum to exactly ``total``.

    Every non-zero entry stays at minimum 1. Zero entries stay zero
    (symbols that never appear cannot be coded). Deterministic — both
    encoder and decoder will derive the identical scaled table from
    the same raw counts.
    """
    total_raw = sum(raw)
    if total_raw == 0:
        raise ValueError("Cannot normalise an all-zero count vector")
    if any(c < 0 for c in raw):
        raise ValueError("counts must be non-negative")

    # First pass: proportional scaling, minimum 1 for non-zero.
    scaled = [
        max(1, (c * total) // total_raw) if c > 0 else 0 for c in raw
    ]

    diff = total - sum(scaled)
    if diff > 0:
        # Add `diff` total to the largest counts.
        order = sorted(range(len(scaled)), key=lambda i: (-scaled[i], i))
        for k in range(diff):
            scaled[order[k % len(order)]] += 1
    elif diff < 0:
        # Subtract from the largest counts that can spare it (must stay >=1
        # for non-zero entries).
        remaining = -diff
        order = sorted(range(len(scaled)), key=lambda i: (-scaled[i], i))
        idx = 0
        guard = 0
        while remaining > 0:
            i = order[idx % len(order)]
            if scaled[i] > 1:
                scaled[i] -= 1
                remaining -= 1
            idx += 1
            guard += 1
            if guard > 100 * len(order):
                break  # shouldn't happen for sane inputs

    if sum(scaled) != total:
        raise AssertionError(
            f"normalise_counts failed to reach {total}; got {sum(scaled)}"
        )
    return scaled


def _build_cumul(freqs: List[int]) -> List[int]:
    cumul = [0] * (len(freqs) + 1)
    s = 0
    for i, f in enumerate(freqs):
        s += f
        cumul[i + 1] = s
    return cumul


def rans_encode(symbols: List[int], raw_freqs: List[int]) -> bytes:
    """Encode ``symbols`` with the static model derived from ``raw_freqs``.

    ``raw_freqs`` is normalised to sum to ``PROB_TOTAL`` internally;
    the caller does not need to pre-normalise.
    """
    if not symbols:
        return b""

    freqs = normalize_counts(raw_freqs, PROB_TOTAL)
    cumul = _build_cumul(freqs)

    state = RANS_BYTE_L
    emitted: List[int] = []  # bytes in emission order (reverse of decode order)

    # Process symbols in reverse so the decoder reads them forward.
    for sym in reversed(symbols):
        freq = freqs[sym]
        if freq == 0:
            raise ValueError(
                f"Symbol {sym} has zero frequency in the model; cannot encode"
            )
        x_max = _RENORM_FACTOR * freq
        # Renormalise out high bytes if encoding would overflow.
        while state >= x_max:
            emitted.append(state & 0xFF)
            state >>= 8
        # Apply the rANS encode step.
        state = ((state // freq) << PROB_BITS) + cumul[sym] + (state % freq)
        if state >= RANS_BYTE_HI:
            raise AssertionError("rANS state overflow")

    # Bytes go to the output in reverse of how they were emitted, then
    # prefixed by the final 4-byte state.
    body = bytes(reversed(emitted))
    return state.to_bytes(4, "little") + body


def rans_decode(blob: bytes, raw_freqs: List[int], n_symbols: int) -> List[int]:
    """Inverse of :func:`rans_encode`.

    Uses the same normalisation of ``raw_freqs`` that the encoder used.
    """
    if n_symbols == 0:
        return []
    if len(blob) < 4:
        raise ValueError("Encoded blob is too short — missing final state")

    freqs = normalize_counts(raw_freqs, PROB_TOTAL)
    cumul = _build_cumul(freqs)
    n_alpha = len(freqs)

    state = int.from_bytes(blob[:4], "little")
    body_pos = 4
    output: List[int] = []

    for _ in range(n_symbols):
        slot = state & (PROB_TOTAL - 1)

        # Find the symbol s with cumul[s] <= slot < cumul[s+1].
        lo, hi = 0, n_alpha - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cumul[mid] <= slot:
                lo = mid
            else:
                hi = mid - 1
        sym = lo

        freq = freqs[sym]
        state = freq * (state >> PROB_BITS) + slot - cumul[sym]

        while state < RANS_BYTE_L:
            if body_pos >= len(blob):
                raise ValueError("Encoded blob exhausted mid-decode")
            state = (state << 8) | blob[body_pos]
            body_pos += 1

        output.append(sym)

    return output
