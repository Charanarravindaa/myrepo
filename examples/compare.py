"""CLI: run Vector-RLE bit-size comparisons.

Usage:
    python examples/compare.py                # built-in synthetic samples
    python examples/compare.py <path-to-file> # benchmark on a real file (bytes)
"""

from __future__ import annotations

import sys
from pathlib import Path

from vrle.bench import _print_row, benchmark, main as run_builtin


def benchmark_file(path: Path) -> None:
    data = list(path.read_bytes())
    name, stats = benchmark(f"file: {path.name}", data, alphabet_size=256)
    _print_row(name, stats)


def main() -> None:
    if len(sys.argv) == 1:
        run_builtin()
        return
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_file():
            print(f"skip (not a file): {p}", file=sys.stderr)
            continue
        benchmark_file(p)


if __name__ == "__main__":
    main()
