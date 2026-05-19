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
from pathlib import Path
from typing import Optional

from .bitpack import BitReader, BitWriter, leb128_decode, leb128_encode
from .bwt import bwt_decode, bwt_encode
from .core import Run, decode_runs, encode_runs
from .deflate_codes import (
    LENGTH_CODES,
    N_DISTANCE_CODES,
    code_to_distance,
    code_to_length,
    distance_extra_bits,
    distance_to_code,
    length_extra_bits,
    length_to_code,
)
from .lz import lz77_decode, lz77_encode
from .mtf import mtf_decode, mtf_encode
from .ppm import (
    decode_ppm_seq,
    decode_ppm_stream,
    encode_ppm_seq,
    encode_ppm_stream,
)
from .shared_dict import SharedDict
from .rangecoder import (
    decode_adaptive_stream as decode_stream,
    encode_adaptive_stream as encode_stream,
)
from .vocab import Vocabulary
from .wordtok import detokenize, tokenize


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


# ---------------------------------------------------------------------------
# Pipeline D: LZ77 + arithmetic coder on four channels
# ---------------------------------------------------------------------------


def _ints_to_leb_bytes(values):
    out = []
    for v in values:
        out.extend(leb128_encode(v))
    return out


def _leb_bytes_to_ints(byte_list, count):
    buf = bytes(byte_list)
    out = []
    pos = 0
    for _ in range(count):
        v, pos = leb128_decode(buf, pos)
        out.append(v)
    return out


def _compress_lz_rc(data: bytes) -> bytes:
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)
    controls, literals, lengths, distances = lz77_encode(data)
    length_bytes = _ints_to_leb_bytes(lengths)
    distance_bytes = _ints_to_leb_bytes(distances)
    out += encode_stream(controls, alphabet_size=2)
    out += encode_stream(literals, alphabet_size=256)
    out += leb128_encode(len(lengths))
    out += encode_stream(length_bytes, alphabet_size=256)
    out += encode_stream(distance_bytes, alphabet_size=256)
    return bytes(out)


def _decompress_lz_rc(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""
    controls, pos = decode_stream(blob, pos)
    literals, pos = decode_stream(blob, pos)
    n_matches, pos = leb128_decode(blob, pos)
    length_bytes, pos = decode_stream(blob, pos)
    distance_bytes, pos = decode_stream(blob, pos)
    lengths = _leb_bytes_to_ints(length_bytes, n_matches)
    distances = _leb_bytes_to_ints(distance_bytes, n_matches)
    out = lz77_decode(controls, literals, lengths, distances)
    assert len(out) == input_len, "lz_rc length mismatch on decode"
    return out


lz_rc = Pipeline("lz_rc", _compress_lz_rc, _decompress_lz_rc)


# ---------------------------------------------------------------------------
# Pipeline E: LZ77 + BWT/MTF/RLE on the literals  (kitchen sink)
# ---------------------------------------------------------------------------
# LZ77 captures word-level matches; whatever literals remain (the "novel"
# characters that didn't fit a back-reference) still have character-level
# entropy that BWT+MTF+RLE can squeeze out. The match streams (controls,
# lengths, distances) are arithmetic-coded directly.


def _compress_lz_bwt_mtf_rle_rc(data: bytes) -> bytes:
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)

    controls, literals, lengths, distances = lz77_encode(data)
    out += encode_stream(controls, alphabet_size=2)

    # Literals through BWT -> MTF -> RLE -> arithmetic.
    out += leb128_encode(len(literals))
    if literals:
        lit_bytes = bytes(literals)
        bwt_out, primary = bwt_encode(lit_bytes)
        mtf_out = mtf_encode(bwt_out)
        out += leb128_encode(primary)
        runs = encode_runs(mtf_out)
        tokens, freq_bytes = _runs_to_streams(runs)
        out += encode_stream(tokens, alphabet_size=256)
        out += encode_stream(freq_bytes, alphabet_size=256)

    # Match streams.
    out += leb128_encode(len(lengths))
    length_bytes = _ints_to_leb_bytes(lengths)
    distance_bytes = _ints_to_leb_bytes(distances)
    out += encode_stream(length_bytes, alphabet_size=256)
    out += encode_stream(distance_bytes, alphabet_size=256)
    return bytes(out)


def _decompress_lz_bwt_mtf_rle_rc(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""

    controls, pos = decode_stream(blob, pos)

    n_literals, pos = leb128_decode(blob, pos)
    if n_literals == 0:
        literals = []
    else:
        primary, pos = leb128_decode(blob, pos)
        tokens, pos = decode_stream(blob, pos)
        freq_bytes, pos = decode_stream(blob, pos)
        runs = _streams_to_runs(tokens, freq_bytes)
        mtf_out = bytes(decode_runs(runs))
        bwt_out = mtf_decode(mtf_out)
        literals = list(bwt_decode(bwt_out, primary))

    n_matches, pos = leb128_decode(blob, pos)
    length_bytes, pos = decode_stream(blob, pos)
    distance_bytes, pos = decode_stream(blob, pos)
    lengths = _leb_bytes_to_ints(length_bytes, n_matches)
    distances = _leb_bytes_to_ints(distance_bytes, n_matches)

    out = lz77_decode(controls, literals, lengths, distances)
    assert len(out) == input_len, "lz_bwt_mtf_rle_rc length mismatch on decode"
    return out


lz_bwt_mtf_rle_rc = Pipeline(
    "lz_bwt_mtf_rle_rc",
    _compress_lz_bwt_mtf_rle_rc,
    _decompress_lz_bwt_mtf_rle_rc,
)


# ---------------------------------------------------------------------------
# Pipeline F: LZ77 with deflate-style bin codes + arithmetic coding
# ---------------------------------------------------------------------------
# This is the part naive `lz_rc` was missing. Lengths and distances are
# emitted as small bin codes (entropy-coded against a tight alphabet) plus
# a few raw extra bits. About 1 byte per match pair instead of ~1.5–2.
# Literals and length codes share one combined alphabet of 285 symbols
# (deflate's literal/length code).
#
# Stream layout in the blob:
#   leb128(input_len)
#   encode_stream(litlen_symbols, alphabet=285)   # literals 0..255, lengths 256..284
#   encode_stream(dist_codes,    alphabet=30)
#   leb128(extras_n_bits) || leb128(extras_byte_len) || extras_bytes


LITLEN_ALPHABET = 256 + len(LENGTH_CODES)  # 256 literals + 29 length codes = 285


def _compress_deflate_rc(data: bytes) -> bytes:
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)

    controls, literals, lengths, distances = lz77_encode(data)
    litlen_symbols: list[int] = []
    dist_codes: list[int] = []
    extras = BitWriter()

    lit_i = 0
    match_i = 0
    for c in controls:
        if c == 0:
            litlen_symbols.append(literals[lit_i])
            lit_i += 1
        else:
            L = lengths[match_i]
            D = distances[match_i]
            match_i += 1
            l_code, l_extra, l_bits = length_to_code(L)
            litlen_symbols.append(256 + l_code)
            if l_bits:
                extras.write_bits(l_extra, l_bits)
            d_code, d_extra, d_bits = distance_to_code(D)
            dist_codes.append(d_code)
            if d_bits:
                extras.write_bits(d_extra, d_bits)

    out += encode_stream(litlen_symbols, alphabet_size=LITLEN_ALPHABET)
    out += encode_stream(dist_codes, alphabet_size=N_DISTANCE_CODES)
    extras_bytes = extras.to_bytes()
    out += leb128_encode(len(extras))           # bit length
    out += leb128_encode(len(extras_bytes))     # byte length
    out += extras_bytes
    return bytes(out)


def _decompress_deflate_rc(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""

    litlen_symbols, pos = decode_stream(blob, pos)
    dist_codes, pos = decode_stream(blob, pos)
    extras_n_bits, pos = leb128_decode(blob, pos)
    extras_byte_len, pos = leb128_decode(blob, pos)
    extras = BitReader(blob[pos : pos + extras_byte_len], extras_n_bits)
    pos += extras_byte_len

    out = bytearray()
    dist_i = 0
    for sym in litlen_symbols:
        if sym < 256:
            out.append(sym)
            continue
        l_code = sym - 256
        l_bits = length_extra_bits(l_code)
        l_extra = extras.read_bits(l_bits) if l_bits else 0
        length = code_to_length(l_code, l_extra)

        d_code = dist_codes[dist_i]
        dist_i += 1
        d_bits = distance_extra_bits(d_code)
        d_extra = extras.read_bits(d_bits) if d_bits else 0
        distance = code_to_distance(d_code, d_extra)

        start = len(out) - distance
        for i in range(length):
            out.append(out[start + i])

    assert len(out) == input_len, "deflate_rc length mismatch on decode"
    return bytes(out)


deflate_rc = Pipeline("deflate_rc", _compress_deflate_rc, _decompress_deflate_rc)


# ---------------------------------------------------------------------------
# Pipeline G: PPM (order-4 context model) directly on the byte stream
# ---------------------------------------------------------------------------
# No LZ77, no BWT — just an adaptive order-N statistical model + arithmetic
# coding. Strongest pipeline on natural text where word-level repetition
# would be sparse but byte-level conditional probabilities are rich.


# ---------------------------------------------------------------------------
# Pipeline I: Pure adaptive arithmetic (order-0) on the raw byte stream
# ---------------------------------------------------------------------------
# No RLE / LZ / BWT / context — just adaptive arithmetic over byte counts.
# Optimal for context-free distributions (e.g. text-like data where each
# byte is drawn from a skewed distribution but successive bytes are
# independent).


def _compress_arith_rc(data: bytes) -> bytes:
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)
    out += encode_stream(list(data), alphabet_size=256)
    return bytes(out)


def _decompress_arith_rc(blob: bytes) -> bytes:
    pos = 0
    n, pos = leb128_decode(blob, pos)
    if n == 0:
        return b""
    syms, _ = decode_stream(blob, pos)
    return bytes(syms)


arith_rc = Pipeline("arith_rc", _compress_arith_rc, _decompress_arith_rc)


def _adaptive_order(n: int) -> int:
    """Pick a PPM order based on input size.

    With exclusion enabled, order 4 is consistently best on inputs
    above ~1 KB. Below that we fall to order 2 because deeper contexts
    barely ever repeat and the model never settles.
    """
    if n < 1000:
        return 2
    return 4


def _compress_ppm_rc(data: bytes) -> bytes:
    return encode_ppm_stream(data, order=_adaptive_order(len(data)))


def _decompress_ppm_rc(blob: bytes) -> bytes:
    out, _ = decode_ppm_stream(blob)
    return out


ppm_rc = Pipeline("ppm_rc", _compress_ppm_rc, _decompress_ppm_rc)


# ---------------------------------------------------------------------------
# Pipeline H: BWT + MTF + PPM (low-order context on the clustered stream)
# ---------------------------------------------------------------------------
# After BWT+MTF the byte stream is heavily skewed toward small values
# (especially 0). A low-order PPM (order 2) on this is often a stronger
# entropy step than the RLE+arith chain.


def _compress_bwt_ppm_rc(data: bytes) -> bytes:
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)
    bwt_out, primary = bwt_encode(data)
    mtf_out = mtf_encode(bwt_out)
    out += leb128_encode(primary)
    out += encode_ppm_stream(mtf_out, order=2)
    return bytes(out)


def _decompress_bwt_ppm_rc(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""
    primary, pos = leb128_decode(blob, pos)
    mtf_out, pos = decode_ppm_stream(blob, pos)
    bwt_out = mtf_decode(mtf_out)
    return bwt_decode(bwt_out, primary)


bwt_ppm_rc = Pipeline("bwt_ppm_rc", _compress_bwt_ppm_rc, _decompress_bwt_ppm_rc)


# ---------------------------------------------------------------------------
# Pipeline J: Geometric tokenization — word-level Vector-RLE + rANS
# ---------------------------------------------------------------------------
# The v5 unification. Tokenise the input into words; build a Vocabulary
# whose magnitude vector serves *double duty* as both the dictionary
# (token <-> id) and the entropy model that rANS uses to code the
# positional index. One geometric object, two roles.


def _vector_rle_ppm_order(n_tokens: int) -> int:
    """PPM order for the positional index, by token count.

    Higher orders catch bigrams / trigrams once we have enough data
    for them to repeat. Tuned to match the byte-PPM heuristics.
    """
    if n_tokens < 200:
        return 0
    if n_tokens < 2000:
        return 1
    return 2


def _compress_vector_rle(data: bytes) -> bytes:
    """Geometric tokenization with word-level PPM throughout.

    The vocabulary's magnitude vector is the dictionary AND the seed for
    PPM's order-0 model. We don't even ship the magnitudes — the PPM
    decoder learns them from the encoded positional stream. The tokens
    themselves are stored in alphabetical (lexicographic) order so the
    encoder and decoder agree on the token <-> ID mapping without any
    frequency table being written.

    Three nested PPM-compressed streams plus a header. Layout:

        leb128(input_len)
        leb128(n_unique_tokens)
        leb128(total_token_count)
        encode_ppm_stream(length_byte_string)        # token lengths (each <=255)
        encode_ppm_stream(concatenated_token_bytes)  # dictionary content
        encode_ppm_seq(positional_ids, n_unique)     # only if n_unique > 1
    """
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)

    tokens = tokenize(data)
    unique_tokens = sorted(set(tokens))  # canonical lexicographic order
    n_unique = len(unique_tokens)
    tok_to_id = {t: i for i, t in enumerate(unique_tokens)}

    out += leb128_encode(n_unique)
    out += leb128_encode(len(tokens))

    lengths_bytes = bytes(len(t) for t in unique_tokens)
    out += encode_ppm_stream(lengths_bytes, order=2)
    out += encode_ppm_stream(b"".join(unique_tokens), order=4)

    if n_unique > 1:
        ids = [tok_to_id[t] for t in tokens]
        order = _vector_rle_ppm_order(len(ids))
        out += encode_ppm_seq(ids, alphabet_size=n_unique, order=order)
    return bytes(out)


def _decompress_vector_rle(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""
    n_unique, pos = leb128_decode(blob, pos)
    total, pos = leb128_decode(blob, pos)

    lengths_bytes, pos = decode_ppm_stream(blob, pos)
    bytes_blob, pos = decode_ppm_stream(blob, pos)

    unique_tokens: list = []
    cursor = 0
    for L in lengths_bytes:
        unique_tokens.append(bytes_blob[cursor : cursor + L])
        cursor += L

    if n_unique <= 1:
        seq_tokens = [unique_tokens[0]] * total if (n_unique == 1 and total > 0) else []
    else:
        ids, _ = decode_ppm_seq(blob, pos)
        seq_tokens = [unique_tokens[i] for i in ids]
    return detokenize(seq_tokens)


vector_rle = Pipeline("vector_rle", _compress_vector_rle, _decompress_vector_rle)


# ---------------------------------------------------------------------------
# Pipeline K: vector_rle with a shared base dictionary
# ---------------------------------------------------------------------------
# Tokens already present in the shared base dict are referenced by their
# base ID and their bytes are NOT shipped. Novel tokens (per-file delta)
# go into the file's local extension dict with IDs starting at N_base.
# Receivers may opportunistically absorb the delta into a local extended
# dict; the file itself stays portable against the shared base.

_BASE_DICT_CACHE: Optional[SharedDict] = None  # type: ignore[name-defined]


def _shared_base_dict() -> SharedDict:
    global _BASE_DICT_CACHE
    if _BASE_DICT_CACHE is None:
        path = Path(__file__).resolve().parent.parent / "examples" / "data" / "base_english.dict"
        _BASE_DICT_CACHE = SharedDict.load(path)
    return _BASE_DICT_CACHE


def _compress_vector_rle_shared(data: bytes) -> bytes:
    out = bytearray()
    out += leb128_encode(len(data))
    if not data:
        return bytes(out)

    base = _shared_base_dict()
    n_base = len(base)

    tokens = tokenize(data)

    # Partition tokens: those in base vs novel. Novel tokens sorted
    # alphabetically for canonical local-ID assignment (no frequency
    # info written; PPM learns it).
    seen_local: set = set()
    novel_tokens: list = []
    for t in tokens:
        if base.get_id(t) is None and t not in seen_local:
            seen_local.add(t)
            novel_tokens.append(t)
    novel_tokens.sort()

    n_local = len(novel_tokens)
    local_id_of: dict = {t: n_base + i for i, t in enumerate(novel_tokens)}

    out += leb128_encode(base.version)
    out += leb128_encode(n_local)
    out += leb128_encode(len(tokens))

    if n_local > 0:
        lengths_bytes = bytes(len(t) for t in novel_tokens)
        out += encode_ppm_stream(lengths_bytes, order=2)
        out += encode_ppm_stream(b"".join(novel_tokens), order=4)

    alphabet = n_base + n_local
    if alphabet > 1:
        ids = [
            base.get_id(t) if base.get_id(t) is not None else local_id_of[t]
            for t in tokens
        ]
        order = _vector_rle_ppm_order(len(ids))
        out += encode_ppm_seq(ids, alphabet_size=alphabet, order=order)
    return bytes(out)


def _decompress_vector_rle_shared(blob: bytes) -> bytes:
    pos = 0
    input_len, pos = leb128_decode(blob, pos)
    if input_len == 0:
        return b""

    base = _shared_base_dict()
    n_base = len(base)

    version, pos = leb128_decode(blob, pos)
    if version != base.version:
        raise ValueError(
            f"shared dict version mismatch: file expects {version}, runtime has {base.version}"
        )
    n_local, pos = leb128_decode(blob, pos)
    total, pos = leb128_decode(blob, pos)

    novel_tokens: list = []
    if n_local > 0:
        lengths_bytes, pos = decode_ppm_stream(blob, pos)
        bytes_blob, pos = decode_ppm_stream(blob, pos)
        cursor = 0
        for L in lengths_bytes:
            novel_tokens.append(bytes_blob[cursor : cursor + L])
            cursor += L

    alphabet = n_base + n_local
    if alphabet <= 1:
        if total == 0:
            return b""
        tok = base.get_token(0) if n_local == 0 else novel_tokens[0]
        return tok * total

    ids, _ = decode_ppm_seq(blob, pos)
    seq_tokens = [
        base.get_token(i) if i < n_base else novel_tokens[i - n_base]
        for i in ids
    ]
    return detokenize(seq_tokens)


vector_rle_shared = Pipeline(
    "vector_rle_shared",
    _compress_vector_rle_shared,
    _decompress_vector_rle_shared,
)


ALL_PIPELINES = [
    rle_rc,
    mtf_rle_rc,
    bwt_mtf_rle_rc,
    lz_rc,
    lz_bwt_mtf_rle_rc,
    deflate_rc,
    arith_rc,
    ppm_rc,
    bwt_ppm_rc,
    vector_rle,
    vector_rle_shared,
]
