"""Train a ``SharedDict`` from a corpus of text files.

Usage:
    python -m vrle.train_dict <corpus_dir-or-file>... -o <output> [-n N]

Tokenises every input file with :mod:`vrle.wordtok`, counts tokens
across the entire corpus, and writes the top-N most frequent tokens
(in descending-frequency order) to the output path.

The order of tokens in the resulting dict is the canonical ID
assignment — every consumer of the dict must read it as-is.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from .shared_dict import SharedDict
from .wordtok import tokenize


def _iter_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for sub in sorted(pp.iterdir()):
                if sub.is_file():
                    yield sub
        elif pp.is_file():
            yield pp


def train(
    inputs: Iterable[str | Path],
    output: str | Path,
    top_n: int = 4000,
    version: int = 1,
) -> SharedDict:
    counts: Counter = Counter()
    n_files = 0
    n_bytes = 0
    for path in _iter_files(inputs):
        data = path.read_bytes()
        n_files += 1
        n_bytes += len(data)
        for tok in tokenize(data):
            counts[tok] += 1
    if not counts:
        raise ValueError("Empty training corpus")
    ordered = [tok for tok, _ in counts.most_common(top_n)]
    d = SharedDict(ordered, version=version)
    d.save(output)
    return d


def _main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Train a vrle SharedDict.")
    p.add_argument("inputs", nargs="+", help="Files or directories.")
    p.add_argument("-o", "--output", required=True, help="Output dict path.")
    p.add_argument("-n", "--top-n", type=int, default=4000)
    p.add_argument("-v", "--version", type=int, default=1)
    args = p.parse_args(argv)
    d = train(args.inputs, args.output, top_n=args.top_n, version=args.version)
    print(
        f"trained {len(d)} tokens (top-{args.top_n}) -> {args.output} "
        f"(version {d.version})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
