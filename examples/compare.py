"""CLI for Vector-RLE benchmark / single-file compression.

Usage:
    python examples/compare.py                  # run the built-in benchmark
    python examples/compare.py <file>           # compare schemes on a file
    python examples/compare.py --pipeline bwt_mtf_rle_rc <file>
                                                # run only one pipeline + decode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vrle.bench import _measure, _print_table, main as run_builtin
from vrle.pipelines import ALL_PIPELINES, Pipeline


def _by_name(name: str) -> Pipeline:
    for p in ALL_PIPELINES:
        if p.name == name:
            return p
    raise SystemExit(f"unknown pipeline {name!r}; choose from {[p.name for p in ALL_PIPELINES]}")


def cmd() -> None:
    parser = argparse.ArgumentParser(description="Vector-RLE benchmark / round-trip check")
    parser.add_argument("file", nargs="*", help="file(s) to benchmark; default = built-in samples")
    parser.add_argument(
        "--pipeline",
        choices=[p.name for p in ALL_PIPELINES],
        help="run a single pipeline (compress + decompress + verify) and report sizes",
    )
    args = parser.parse_args()

    if not args.file:
        if args.pipeline:
            parser.error("--pipeline requires a file argument")
        run_builtin()
        return

    for path_str in args.file:
        path = Path(path_str)
        if not path.is_file():
            print(f"skip (not a file): {path}", file=sys.stderr)
            continue
        data = path.read_bytes()
        if args.pipeline:
            p = _by_name(args.pipeline)
            blob = p.compress(data)
            decoded = p.decompress(blob)
            ok = "ok" if decoded == data else "MISMATCH"
            print(f"{path}: raw={len(data)} -> {p.name}={len(blob)} ({100*len(blob)/max(1,len(data)):.2f}% raw)  [{ok}]")
        else:
            sizes = _measure(str(path), data)
            _print_table(f"file: {path.name}", sizes)


if __name__ == "__main__":
    cmd()
