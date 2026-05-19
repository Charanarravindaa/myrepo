"""Variable-length integer encodings and bit-level packing helpers.

Two frequency encodings are provided so the benchmark can compare them:

  * LEB128 — byte-aligned, 7 value bits per byte. Simple, ubiquitous.
  * Elias gamma — bit-level, optimal for small magnitudes (1 -> 1 bit,
    2..3 -> 3 bits, 4..7 -> 5 bits, ...). Lines up naturally with the
    "magnitude" framing of Vector-RLE.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from .core import Run


# ---------------------------------------------------------------------------
# LEB128 (byte-aligned variable-length unsigned int)
# ---------------------------------------------------------------------------


def leb128_encode(n: int) -> bytes:
    if n < 0:
        raise ValueError("LEB128 unsigned encoder requires n >= 0")
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def leb128_decode(buf: bytes, pos: int = 0) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def leb128_bits(n: int) -> int:
    """Bit-cost of writing n as LEB128 (always a whole number of bytes)."""
    return len(leb128_encode(n)) * 8


# ---------------------------------------------------------------------------
# Elias gamma (bit-level variable-length encoding of positive ints)
# ---------------------------------------------------------------------------


def gamma_bits(n: int) -> int:
    if n <= 0:
        raise ValueError("Elias gamma encodes positive integers only")
    width = n.bit_length()  # number of bits to represent n in binary
    return 2 * width - 1  # (width-1) zero prefix + width binary bits


class BitWriter:
    def __init__(self) -> None:
        self._bits: List[int] = []

    def write_bit(self, b: int) -> None:
        self._bits.append(b & 1)

    def write_bits(self, value: int, width: int) -> None:
        for i in range(width - 1, -1, -1):
            self._bits.append((value >> i) & 1)

    def write_gamma(self, n: int) -> None:
        if n <= 0:
            raise ValueError("Elias gamma encodes positive integers only")
        width = n.bit_length()
        for _ in range(width - 1):
            self._bits.append(0)
        self.write_bits(n, width)

    def __len__(self) -> int:
        return len(self._bits)

    def to_bytes(self) -> bytes:
        out = bytearray((len(self._bits) + 7) // 8)
        for i, bit in enumerate(self._bits):
            if bit:
                out[i >> 3] |= 1 << (7 - (i & 7))
        return bytes(out)


class BitReader:
    def __init__(self, data: bytes, bit_length: int) -> None:
        self._data = data
        self._bit_length = bit_length
        self._pos = 0

    def read_bit(self) -> int:
        if self._pos >= self._bit_length:
            raise EOFError("BitReader exhausted")
        byte = self._data[self._pos >> 3]
        bit = (byte >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    def read_bits(self, width: int) -> int:
        v = 0
        for _ in range(width):
            v = (v << 1) | self.read_bit()
        return v

    def read_gamma(self) -> int:
        zeros = 0
        while self.read_bit() == 0:
            zeros += 1
        # leading '1' already consumed; read remaining `zeros` bits
        tail = self.read_bits(zeros) if zeros else 0
        return (1 << zeros) | tail


# ---------------------------------------------------------------------------
# Run packing
# ---------------------------------------------------------------------------


def pack_runs(
    runs: Iterable[Run],
    token_width: int,
    freq_encoding: str = "leb128",
) -> Tuple[bytes, int]:
    """Pack runs into a byte buffer.

    Tokens are written at a fixed bit width (caller's responsibility to make
    sure every token fits — typically derived from the alphabet size).
    Frequencies use the chosen variable-length encoding.

    Returns (buffer, total_bit_length).
    """
    if freq_encoding not in ("leb128", "gamma", "raw32"):
        raise ValueError(f"unknown freq_encoding {freq_encoding!r}")

    writer = BitWriter()
    for run in runs:
        token_int = int(run.token)
        if token_int < 0 or token_int >> token_width:
            raise ValueError(
                f"token {token_int} does not fit in {token_width} bits"
            )
        writer.write_bits(token_int, token_width)
        if freq_encoding == "leb128":
            for byte in leb128_encode(run.freq):
                writer.write_bits(byte, 8)
        elif freq_encoding == "gamma":
            writer.write_gamma(run.freq)
        else:  # raw32
            writer.write_bits(run.freq, 32)
    return writer.to_bytes(), len(writer)


def unpack_runs(
    data: bytes,
    bit_length: int,
    n_runs: int,
    token_width: int,
    freq_encoding: str = "leb128",
) -> List[Run]:
    """Inverse of pack_runs. Caller must supply n_runs (no length prefix)."""
    reader = BitReader(data, bit_length)
    out: List[Run] = []
    for _ in range(n_runs):
        token = reader.read_bits(token_width)
        if freq_encoding == "leb128":
            shift = 0
            result = 0
            while True:
                byte = reader.read_bits(8)
                result |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            freq = result
        elif freq_encoding == "gamma":
            freq = reader.read_gamma()
        elif freq_encoding == "raw32":
            freq = reader.read_bits(32)
        else:
            raise ValueError(f"unknown freq_encoding {freq_encoding!r}")
        out.append(Run(token, freq))
    return out
