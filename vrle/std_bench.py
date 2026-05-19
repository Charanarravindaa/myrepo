"""Standard-corpus benchmark: vector_auto vs gzip/bz2/zstd/brotli.

Runs on a held-out set of public-domain English literature
(NLTK Gutenberg + Brown), with the shared dictionary trained on a
disjoint subset of the same corpora.

Truncates each file to a fixed cap to keep pure-Python runtime
bounded; the cap is chosen so the corpus is real-size for compression
purposes (each test file is large enough to amortise model warmup).
"""

from __future__ import annotations

import bz2
import gzip
import time
from pathlib import Path
from typing import List, Tuple

CAP_PER_FILE = 200_000  # 200 KB per file is plenty for compressor settle


def _baselines() -> List[Tuple[str, "callable"]]:
    out = [
        ("gzip(9)", lambda d: gzip.compress(d, compresslevel=9)),
        ("bz2(9)", lambda d: bz2.compress(d, compresslevel=9)),
    ]
    try:
        import zstandard as zs
        out.append(("zstd(22)", lambda d: zs.ZstdCompressor(level=22).compress(d)))
    except ImportError:
        pass
    try:
        import brotli as br
        out.append(("brotli(11)", lambda d: br.compress(d, quality=11)))
    except ImportError:
        pass
    return out


def _load_corpus(corpus_dir: Path) -> List[Tuple[str, bytes]]:
    files: List[Tuple[str, bytes]] = []
    for p in sorted(corpus_dir.iterdir()):
        if not p.is_file():
            continue
        data = p.read_bytes()
        if not data:
            continue
        if len(data) > CAP_PER_FILE:
            data = data[:CAP_PER_FILE]
        files.append((p.name, data))
    return files


def main(corpus_dir: str = "/tmp/std_corpus/test") -> None:
    # Late import so we pick up dict-swap if any.
    from .pipelines import _NAMED_DICT_CACHE, vector_auto
    from .shared_dict import SharedDict

    # Use the Gutenberg-trained dict (held out from the test set).
    dict_path = Path(__file__).resolve().parent / "dicts" / "english_v2.dict"
    if dict_path.exists():
        _NAMED_DICT_CACHE["english"] = SharedDict.load(dict_path)

    base = _NAMED_DICT_CACHE.get("english")
    if base is None:
        from .pipelines import _load_named_dict
        base = _load_named_dict("english")
    print(f"Shared dict: {len(base)} tokens, version {base.version}")

    corpus = _load_corpus(Path(corpus_dir))
    if not corpus:
        raise SystemExit(f"no files in {corpus_dir}")

    baselines = _baselines()
    print(f"Corpus: {len(corpus)} files, {sum(len(d) for _, d in corpus):,} bytes")
    print(f"Baselines: {[name for name, _ in baselines]}")
    print()

    schemes = [("vector_auto", vector_auto.compress)] + baselines
    print(f"{'file':<28} {'raw':>9} | "
          + " | ".join(f"{name:>11}" for name, _ in schemes))
    print("-" * (28 + 9 + 3 + (12 * len(schemes))))

    totals = {name: 0 for name, _ in schemes}
    totals_raw = 0

    for name, data in corpus:
        totals_raw += len(data)
        row = []
        for sname, fn in schemes:
            t0 = time.perf_counter()
            blob = fn(data)
            totals[sname] += len(blob)
            row.append(f"{len(blob):>11,}")
        short = name[:26] + ("..." if len(name) > 28 else "")
        print(f"{short:<28} {len(data):>9,} | " + " | ".join(row))

    print("-" * (28 + 9 + 3 + (12 * len(schemes))))
    print(f"{'TOTAL':<28} {totals_raw:>9,} | "
          + " | ".join(f"{totals[s]:>11,}" for s, _ in schemes))
    print(f"{'% of raw':<28} {'100.00%':>9} | "
          + " | ".join(f"{100*totals[s]/totals_raw:>10.2f}%" for s, _ in schemes))
    base_total = totals["gzip(9)"]
    print(f"{'% of gzip':<28} {'':>9} | "
          + " | ".join(
              f"{100*totals[s]/base_total:>10.2f}%" if base_total else "  n/a"
              for s, _ in schemes
          ))


if __name__ == "__main__":
    main()
