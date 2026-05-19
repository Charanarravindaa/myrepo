"""Head-to-head compression benchmark.

Compares the three Vector-RLE pipelines against:
  * raw bytes
  * gzip      (stdlib)
  * bz2       (stdlib)
  * zstd      (optional dev dep)
  * brotli    (optional dev dep)

Every pipeline must round-trip; the harness asserts it and aborts loudly if
not. Speed is intentionally not measured — these pipelines are Python and
will lose every speed race; the interesting axis is bit ratio.

Run with: ``python -m vrle.bench``
"""

from __future__ import annotations

import bz2
import gzip
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .pipelines import ALL_PIPELINES, Pipeline

try:  # optional
    import zstandard as _zstd  # type: ignore

    def _zstd_compress(data: bytes) -> bytes:
        return _zstd.ZstdCompressor(level=22).compress(data)

except Exception:  # pragma: no cover - optional dep
    _zstd_compress = None  # type: ignore

try:  # optional
    import brotli as _brotli  # type: ignore

    def _brotli_compress(data: bytes) -> bytes:
        return _brotli.compress(data, quality=11)

except Exception:  # pragma: no cover - optional dep
    _brotli_compress = None  # type: ignore


Baseline = Tuple[str, Callable[[bytes], bytes]]


def _baselines() -> List[Baseline]:
    out: List[Baseline] = [
        ("gzip(9)", lambda d: gzip.compress(d, compresslevel=9)),
        ("bz2(9)", lambda d: bz2.compress(d, compresslevel=9)),
    ]
    if _zstd_compress is not None:
        out.append(("zstd(22)", _zstd_compress))
    else:
        out.append(("zstd", None))  # type: ignore
    if _brotli_compress is not None:
        out.append(("brotli(11)", _brotli_compress))
    else:
        out.append(("brotli", None))  # type: ignore
    return out


def _measure(name: str, data: bytes) -> Dict[str, int]:
    sizes: Dict[str, int] = {"raw": len(data)}
    for p in ALL_PIPELINES:
        blob = p.compress(data)
        roundtrip = p.decompress(blob)
        if roundtrip != data:
            raise AssertionError(f"pipeline {p.name!r} failed round-trip on {name!r}")
        sizes[p.name] = len(blob)
    for label, fn in _baselines():
        if fn is None:
            sizes[label] = -1  # marker for "not installed"
        else:
            sizes[label] = len(fn(data))
    return sizes


def _print_table(name: str, sizes: Dict[str, int]) -> None:
    raw = sizes["raw"]
    print(f"\n== {name} ==  ({raw} bytes raw)")
    print(f"  {'scheme':<18} {'bytes':>10}   {'% raw':>7}   {'% gzip':>7}")
    gz = sizes.get("gzip(9)", -1)
    for label, size in sizes.items():
        if size < 0:
            print(f"  {label:<18} {'(not installed)':>10}")
            continue
        pct_raw = f"{100 * size / raw:6.2f}%" if raw else "    n/a"
        pct_gz = (
            f"{100 * size / gz:6.2f}%" if gz > 0 and label != "gzip(9)" else ""
        )
        marker = "  <"  # winners get marked below in the summary
        print(f"  {label:<18} {size:>10}   {pct_raw:>7}   {pct_gz:>7}")
    # Summary line — the smallest scheme.
    best = min(
        (sz, label) for label, sz in sizes.items() if sz > 0 and label != "raw"
    )
    print(f"  winner: {best[1]} ({best[0]} bytes)")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _rand_bytes(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(n))


def _long_runs(n: int, alphabet: int = 8, seed: int = 2) -> bytes:
    rng = random.Random(seed)
    out = bytearray()
    while len(out) < n:
        tok = rng.randrange(alphabet)
        run = rng.randint(20, 200)
        out.extend([tok] * min(run, n - len(out)))
    return bytes(out)


def _text_like(n: int, seed: int = 3) -> bytes:
    rng = random.Random(seed)
    # English-letter-like frequencies, lowercase + space.
    population = list(range(ord("a"), ord("a") + 26)) + [ord(" ")]
    weights = [
        8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.2, 0.8,
        4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0,
        2.4, 0.2, 2.0, 0.1, 18.3,
    ]
    return bytes(rng.choices(population, weights=weights, k=n))


def _log_stream(n_lines: int = 500, seed: int = 4) -> bytes:
    rng = random.Random(seed)
    lines = []
    methods = ["GET", "POST", "PUT", "GET", "GET", "GET"]
    paths = ["/", "/index.html", "/api/v1/users", "/static/app.css", "/favicon.ico"]
    statuses = [200, 200, 200, 200, 200, 301, 404, 500]
    for _ in range(n_lines):
        lines.append(
            f"{rng.choice(methods)} {rng.choice(paths)} HTTP/1.1 {rng.choice(statuses)}\n"
        )
    return "".join(lines).encode()


def _load_sample_text() -> bytes:
    p = Path(__file__).resolve().parent.parent / "examples" / "data" / "sample.txt"
    if p.is_file():
        return p.read_bytes()
    return b""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("Vector-RLE v2 benchmark")
    print("=======================")
    print(
        "Pipelines: "
        + ", ".join(p.name for p in ALL_PIPELINES)
        + "; baselines: raw, gzip, bz2, zstd, brotli"
    )

    datasets: List[Tuple[str, bytes]] = [
        ("Random bytes (incompressible)", _rand_bytes(20_000, 1)),
        ("High-redundancy bytes (long runs)", _long_runs(20_000)),
        ("English-letter-frequency text", _text_like(20_000)),
        ("Synthetic web log stream", _log_stream(500)),
    ]
    real_text = _load_sample_text()
    if real_text:
        datasets.append(("Real English prose (Pride & Prejudice excerpt)", real_text))

    for name, data in datasets:
        sizes = _measure(name, data)
        _print_table(name, sizes)


if __name__ == "__main__":
    main()
