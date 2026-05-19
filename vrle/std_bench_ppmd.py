"""Standard-corpus benchmark including PPMd (Shkarin's PPMII) at -mx9 -mmem=2g.

PPMd is the production-quality reference PPM implementation that ships
inside 7-zip. Running it as a baseline against ``vector_auto`` is the
honest comparison: our pipeline is PPM-class, so PPMd is the algorithm
most likely to beat us on text.
"""

from __future__ import annotations

import bz2
import gzip
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

CAP_PER_FILE = 200_000


def _ppmd_compress(data: bytes) -> bytes:
    """Compress with 7z's PPMd codec at maximum settings."""
    with tempfile.TemporaryDirectory() as td:
        input_path = Path(td) / "in.bin"
        output_path = Path(td) / "out.7z"
        input_path.write_bytes(data)
        subprocess.run(
            [
                "7z",
                "a",
                "-bd",       # no progress
                "-m0=PPMd",  # PPMd codec
                "-mx=9",     # max level
                "-mmem=2g",  # 2 GB model memory
                "-mo=8",     # order 8 (PPMd parameter)
                str(output_path),
                str(input_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_path.read_bytes()


def _baselines() -> List[Tuple[str, "callable"]]:
    out: List[Tuple[str, callable]] = [
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
    if os.access("/usr/bin/7z", os.X_OK):
        out.append(("PPMd-mx9", _ppmd_compress))
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
    from .pipelines import _NAMED_DICT_CACHE, vector_auto
    from .shared_dict import SharedDict

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
    print(f"{'file':<26}{'raw':>9} | "
          + " | ".join(f"{name:>10}" for name, _ in schemes))
    print("-" * (26 + 9 + 3 + (13 * len(schemes))))

    totals = {name: 0 for name, _ in schemes}
    totals_raw = 0

    for name, data in corpus:
        totals_raw += len(data)
        row = []
        for sname, fn in schemes:
            blob = fn(data)
            totals[sname] += len(blob)
            row.append(f"{len(blob):>10,}")
        short = name[:24]
        print(f"{short:<26}{len(data):>9,} | " + " | ".join(row))

    print("-" * (26 + 9 + 3 + (13 * len(schemes))))
    print(f"{'TOTAL':<26}{totals_raw:>9,} | "
          + " | ".join(f"{totals[s]:>10,}" for s, _ in schemes))
    print(f"{'% of raw':<26}{'100.00%':>9} | "
          + " | ".join(f"{100*totals[s]/totals_raw:>9.2f}%" for s, _ in schemes))
    base_total = totals["gzip(9)"]
    if base_total:
        print(f"{'% of gzip':<26}{'':>9} | "
              + " | ".join(
                  f"{100*totals[s]/base_total:>9.2f}%" for s, _ in schemes
              ))


if __name__ == "__main__":
    main()
