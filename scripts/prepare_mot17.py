"""Creates the half1 / half2 / full MOT17 evaluation splits under data/mot17."""

import argparse
from pathlib import Path

from visiontrack.eval.mot17 import prepare_splits

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/MOT17/train"))
    ap.add_argument("--dst", type=Path, default=Path("data/mot17"))
    args = ap.parse_args()
    splits = prepare_splits(args.src, args.dst)
    for name, seqs in splits.items():
        total = sum(s.length for s in seqs)
        print(f"{name:6s} {len(seqs)} sequences, {total} frames -> {args.dst / name}")
