"""Bit-usage benchmarks across raw / plain-RLE / Vector-RLE representations.

Run with: ``python -m vrle.bench``
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Tuple

from .mathvec import MathRLE
from .sequence import SequenceRLE


def _token_width(alphabet_size: int) -> int:
    if alphabet_size <= 1:
        return 1
    return max(1, math.ceil(math.log2(alphabet_size)))


def _raw_bits(data: Iterable[int], token_width: int) -> int:
    # consume into a length without materialising twice
    n = sum(1 for _ in data)
    return n * token_width


def _plain_rle_bits(seq: SequenceRLE, token_width: int) -> int:
    """Fixed-width baseline: token_width + 32-bit unsigned freq per run."""
    return seq.bit_size(token_width, freq_encoding="raw32")


def benchmark(name: str, data: List[int], alphabet_size: int) -> Tuple[str, dict]:
    token_width = _token_width(alphabet_size)
    seq = SequenceRLE.from_iterable(data)
    mv = MathRLE.from_iterable(data)

    raw = len(data) * token_width
    plain = _plain_rle_bits(seq, token_width)
    seq_leb = seq.bit_size(token_width, "leb128")
    seq_gamma = seq.bit_size(token_width, "gamma")
    mv_leb = mv.bit_size(token_width, "leb128")
    mv_gamma = mv.bit_size(token_width, "gamma")

    return name, {
        "tokens": len(data),
        "alphabet": alphabet_size,
        "token_width": token_width,
        "runs": len(seq.runs),
        "raw_bits": raw,
        "plain_rle_bits": plain,
        "seq_rle_leb128_bits": seq_leb,
        "seq_rle_gamma_bits": seq_gamma,
        "mathvec_leb128_bits": mv_leb,
        "mathvec_gamma_bits": mv_gamma,
    }


def _print_row(name: str, stats: dict) -> None:
    raw = stats["raw_bits"]

    def pct(bits: int) -> str:
        if raw == 0:
            return "  n/a"
        return f"{100 * bits / raw:5.1f}%"

    print(f"\n== {name} ==")
    print(
        f"  tokens={stats['tokens']:>7}  alphabet={stats['alphabet']:>4}  "
        f"runs={stats['runs']:>6}  token_width={stats['token_width']} bits"
    )
    print(f"  {'raw':<22} {raw:>10} bits   ({pct(raw)} of raw)")
    print(
        f"  {'plain RLE (raw32)':<22} {stats['plain_rle_bits']:>10} bits   "
        f"({pct(stats['plain_rle_bits'])} of raw)"
    )
    print(
        f"  {'seq-RLE (LEB128)':<22} {stats['seq_rle_leb128_bits']:>10} bits   "
        f"({pct(stats['seq_rle_leb128_bits'])} of raw)"
    )
    print(
        f"  {'seq-RLE (gamma)':<22} {stats['seq_rle_gamma_bits']:>10} bits   "
        f"({pct(stats['seq_rle_gamma_bits'])} of raw)"
    )
    print(
        f"  {'mathvec (LEB128)':<22} {stats['mathvec_leb128_bits']:>10} bits   "
        f"({pct(stats['mathvec_leb128_bits'])} of raw)  [order lost]"
    )
    print(
        f"  {'mathvec (gamma)':<22} {stats['mathvec_gamma_bits']:>10} bits   "
        f"({pct(stats['mathvec_gamma_bits'])} of raw)  [order lost]"
    )


def _gen_random_bytes(n: int, seed: int = 1) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(256) for _ in range(n)]


def _gen_long_runs(n: int, alphabet: int = 8, seed: int = 2) -> List[int]:
    """Repetitive data with long runs — RLE's best case."""
    rng = random.Random(seed)
    out: List[int] = []
    while len(out) < n:
        tok = rng.randrange(alphabet)
        run_len = rng.randint(20, 200)
        out.extend([tok] * min(run_len, n - len(out)))
    return out


def _gen_text_like(n: int, seed: int = 3) -> List[int]:
    """A mid-redundancy stream (English-letter-like distribution, short runs)."""
    rng = random.Random(seed)
    # Skewed alphabet of 27 tokens (a-z + space)
    weights = [
        8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.2, 0.8,
        4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0,
        2.4, 0.2, 2.0, 0.1, 18.3,
    ]
    population = list(range(27))
    return rng.choices(population, weights=weights, k=n)


def _gen_log_lines(n_lines: int = 200) -> List[int]:
    """A repetitive token stream like web-server logs (status code repeated)."""
    rng = random.Random(4)
    stream: List[int] = []
    statuses = [200, 200, 200, 200, 200, 200, 200, 301, 404, 500]
    for _ in range(n_lines):
        run = rng.randint(5, 50)
        s = rng.choice(statuses)
        stream.extend([s] * run)
    return stream


def main() -> None:
    print("Vector-RLE bit-usage benchmark")
    print("==============================")

    for name, data, alphabet in [
        ("Random bytes (worst case, no runs)", _gen_random_bytes(20_000), 256),
        ("High-redundancy bytes (long runs)", _gen_long_runs(20_000, 8), 8),
        ("Text-like English chars", _gen_text_like(20_000), 27),
        ("Repetitive log-status stream", _gen_log_lines(500), 600),
    ]:
        _, stats = benchmark(name, data, alphabet)
        _print_row(name, stats)

    print()
    print("Notes:")
    print("  - 'plain RLE' uses fixed-width tokens + 32-bit unsigned freq.")
    print("  - 'seq-RLE' (LEB128/gamma) is the Vector-RLE sequence form.")
    print("  - 'mathvec' discards positional order (counts per token only).")


if __name__ == "__main__":
    main()
