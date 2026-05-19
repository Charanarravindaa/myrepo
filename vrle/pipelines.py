"""Hybrid compression pipelines that build on the v1 Vector-RLE core.

Three pipelines are exposed, in order of increasing ambition:

* ``rle_rc`` — Vector-RLE + arithmetic-coded tokens/frequencies. Smallest
  delta from v1: pure entropy upgrade.
* ``mtf_rle_rc`` — Move-To-Front pre-pass then ``rle_rc``. MTF concentrates
  recently-seen bytes near zero, lengthening runs.
* ``bwt_mtf_rle_rc`` — Burrows-Wheeler pre-pass, then MTF, then RLE +
  arithmetic coding. The classic bzip2 stack with Vector-RLE inside.

Each pipeline exposes ``compress(data: bytes) -> bytes`` and
``decompress(blob: bytes) -> bytes`` and round-trips losslessly,
preserving positional order of the input.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bitpack import leb128_decode, leb128_encode
from .bwt import bwt_decode, bwt_encode
from .core import Run, decode_runs, encode_runs
from .mtf import mtf_decode, mtf_encode
from .rangecoder import decode_stream, encode_stream


def _runs_to_streams(runs):
    tokens = [r.token for r in runs]
    freq_bytes = []
    for r in runs:
        freq_bytes.extend(leb128_encode(r.freq))
    return tokens, freq_bytes


def _streams_to_runs(tokens, freq_bytes):
    out = []
    buf = bytes(freq_bytes)
    pos = 0
    for t in tokens:
        freq, pos = leb128_decode(buf, pos)
        out.append(Run(t, freq))
    return out


# ---------------------------------------------------------------------------
# Pipeline A: RLE + arithmetic coder on two channels
# ---------------------------------------------------------------------------


def _compress_rle_rc(data: bytes) -> bytes:
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)
    runs = encode_runs(data)
    tokens, freq_bytes = _runs_to_streams(runs)
    token_blob = encode_stream(tokens, alphabet_size=256)
    freq_blob = encode_stream(freq_bytes, alphabet_size=256)
    out += leb128_encode(len(token_blob))
    out += token_blob
    out += leb128_encode(len(freq_blob))
    out += freq_blob
    return bytes(out)


def _decompress_rle_rc(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""
    token_len, pos = leb128_decode(blob, pos)
    tokens, _ = decode_stream(blob[pos : pos + token_len])
    pos += token_len
    freq_len, pos = leb128_decode(blob, pos)
    freq_bytes, _ = decode_stream(blob[pos : pos + freq_len])
    pos += freq_len
    runs = _streams_to_runs(tokens, freq_bytes)
    out = bytes(decode_runs(runs))
    assert len(out) == input_len, "rle_rc length mismatch on decode"
    return out


# ---------------------------------------------------------------------------
# Pipeline B: MTF + Pipeline A
# ---------------------------------------------------------------------------


def _compress_mtf_rle_rc(data: bytes) -> bytes:
    if not data:
        return _compress_rle_rc(data)
    transformed = mtf_encode(data)
    return _compress_rle_rc(transformed)


def _decompress_mtf_rle_rc(blob: bytes) -> bytes:
    transformed = _decompress_rle_rc(blob)
    if not transformed:
        return b""
    return mtf_decode(transformed)


# ---------------------------------------------------------------------------
# Pipeline C: BWT + MTF + RLE + arithmetic coder  (bzip2-style stack)
# ---------------------------------------------------------------------------


def _compress_bwt_mtf_rle_rc(data: bytes) -> bytes:
    if not data:
        # Empty: 0-length marker + sentinel primary index of 0.
        out = bytearray()
        out += leb128_encode(0)  # input length
        return bytes(out)
    bwt_out, primary = bwt_encode(data)
    mtf_out = mtf_encode(bwt_out)

    header = bytearray()
    header += leb128_encode(len(data))
    header += leb128_encode(primary)

    runs = encode_runs(mtf_out)
    tokens, freq_bytes = _runs_to_streams(runs)
    token_blob = encode_stream(tokens, alphabet_size=256)
    freq_blob = encode_stream(freq_bytes, alphabet_size=256)

    body = bytearray()
    body += leb128_encode(len(token_blob))
    body += token_blob
    body += leb128_encode(len(freq_blob))
    body += freq_blob
    return bytes(header) + bytes(body)


def _decompress_bwt_mtf_rle_rc(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""
    primary, pos = leb128_decode(blob, pos)

    token_len, pos = leb128_decode(blob, pos)
    tokens, _ = decode_stream(blob[pos : pos + token_len])
    pos += token_len
    freq_len, pos = leb128_decode(blob, pos)
    freq_bytes, _ = decode_stream(blob[pos : pos + freq_len])
    pos += freq_len

    runs = _streams_to_runs(tokens, freq_bytes)
    mtf_out = bytes(decode_runs(runs))
    bwt_out = mtf_decode(mtf_out)
    return bwt_decode(bwt_out, primary)


# ---------------------------------------------------------------------------
# Public interface — light dataclass wrappers so each pipeline can be passed
# around like a value.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pipeline:
    name: str
    compress: "callable"
    decompress: "callable"


rle_rc = Pipeline("rle_rc", _compress_rle_rc, _decompress_rle_rc)
mtf_rle_rc = Pipeline("mtf_rle_rc", _compress_mtf_rle_rc, _decompress_mtf_rle_rc)
bwt_mtf_rle_rc = Pipeline(
    "bwt_mtf_rle_rc", _compress_bwt_mtf_rle_rc, _decompress_bwt_mtf_rle_rc
)

ALL_PIPELINES = [rle_rc, mtf_rle_rc, bwt_mtf_rle_rc]
