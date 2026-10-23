"""MOT17 helpers: per-sequence half splits and ground-truth parsing.

MOT17 only publishes ground truth for its 7 training sequences, so (like
ByteTrack and CenterTrack) each sequence is split in time:

* ``half1``: frames 1 .. floor(N/2)  -> used for every tuning sweep
* ``half2``: frames floor(N/2)+1 .. N -> held-out evaluation
* ``full``:  all frames (reference only)

The detector is never trained on MOT17 here (COCO-pretrained weights), so
both halves are unseen by the model; the split exists to keep tracker and
threshold tuning away from the reported numbers.

Split sequences are materialised as MOTChallenge-style folders (symlinked,
renumbered images + renumbered gt.txt + seqinfo.ini) so that TrackEval and
the video reader treat them like any other sequence.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEQUENCES = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13")

# Qualitative conditions per sequence (from the MOT16/17 paper and visual inspection),
# used to group results in the robustness analysis.
SEQUENCE_CONDITIONS: dict[str, dict] = {
    "MOT17-02": {
        "camera": "static",
        "view": "eye-level",
        "light": "day",
        "density": "medium",
        "notes": "busy square, many partial occlusions",
    },
    "MOT17-04": {
        "camera": "static",
        "view": "elevated",
        "light": "night (street lit)",
        "density": "high",
        "notes": "crowded pedestrian street seen from above",
    },
    "MOT17-05": {
        "camera": "moving",
        "view": "eye-level",
        "light": "day",
        "density": "low",
        "notes": "640x480 walking camera, small/low-res people, 14 FPS",
    },
    "MOT17-09": {
        "camera": "static",
        "view": "eye-level",
        "light": "day",
        "density": "medium",
        "notes": "pedestrian area, close-up people with heavy occlusion",
    },
    "MOT17-10": {
        "camera": "moving",
        "view": "eye-level",
        "light": "night",
        "density": "medium",
        "notes": "low light, motion blur from the moving camera",
    },
    "MOT17-11": {
        "camera": "moving",
        "view": "eye-level",
        "light": "indoor",
        "density": "medium",
        "notes": "shopping mall, forward-moving camera",
    },
    "MOT17-13": {
        "camera": "moving",
        "view": "elevated (bus)",
        "light": "day",
        "density": "medium",
        "notes": "camera on a turning bus: fast apparent motion, small distant people",
    },
}

# MOT17 gt.txt classes: 1 pedestrian, 2 person on vehicle, 7 static person,
# 8 distractor, 12 reflection -> "person-like but not evaluated".
PEDESTRIAN = 1
DISTRACTOR_CLASSES = (2, 7, 8, 12)


@dataclass
class SeqInfo:
    name: str
    path: Path  # sequence folder (contains img1/, gt/, seqinfo.ini)
    length: int
    fps: float
    width: int
    height: int

    @property
    def img_dir(self) -> Path:
        return self.path / "img1"

    @property
    def gt_path(self) -> Path:
        return self.path / "gt" / "gt.txt"


def read_seqinfo(seq_dir: Path) -> SeqInfo:
    cp = configparser.ConfigParser()
    cp.read(seq_dir / "seqinfo.ini")
    s = cp["Sequence"]
    return SeqInfo(
        name=seq_dir.name,
        path=seq_dir,
        length=int(s["seqLength"]),
        fps=float(s["frameRate"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
    )


def load_gt(path: Path) -> np.ndarray:
    """Rows: frame, id, x, y, w, h, consider, class, visibility (float array)."""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    return arr.reshape(-1, 9)


def split_ranges(length: int) -> dict[str, tuple[int, int]]:
    half = length // 2
    return {"half1": (1, half), "half2": (half + 1, length), "full": (1, length)}


def prepare_splits(src_root: Path, dst_root: Path, variant: str = "FRCNN") -> dict[str, list[SeqInfo]]:
    """Builds data/mot17/{half1,half2,full}/MOT17-XX folders from the official train folders."""
    out: dict[str, list[SeqInfo]] = {"half1": [], "half2": [], "full": []}
    for seq in SEQUENCES:
        src = src_root / f"{seq}-{variant}"
        if not src.exists():
            raise FileNotFoundError(f"missing {src}; run scripts/download_data.sh mot17")
        info = read_seqinfo(src)
        gt = load_gt(src / "gt" / "gt.txt")
        images = sorted((src / "img1").glob("*.jpg"))
        if len(images) != info.length:
            raise RuntimeError(f"{seq}: seqinfo says {info.length} frames, found {len(images)}")
        for split, (a, b) in split_ranges(info.length).items():
            dst = dst_root / split / seq
            (dst / "img1").mkdir(parents=True, exist_ok=True)
            (dst / "gt").mkdir(parents=True, exist_ok=True)
            for new_idx, frame in enumerate(range(a, b + 1), start=1):
                link = dst / "img1" / f"{new_idx:06d}.jpg"
                if not link.exists():
                    os.symlink(images[frame - 1].resolve(), link)
            rows = gt[(gt[:, 0] >= a) & (gt[:, 0] <= b)].copy()
            rows[:, 0] -= a - 1
            np.savetxt(
                dst / "gt" / "gt.txt",
                rows,
                delimiter=",",
                fmt=["%d", "%d", "%d", "%d", "%d", "%d", "%d", "%d", "%.5f"],
            )
            n = b - a + 1
            (dst / "seqinfo.ini").write_text(
                "[Sequence]\n"
                f"name={seq}\nimDir=img1\nframeRate={info.fps:g}\nseqLength={n}\n"
                f"imWidth={info.width}\nimHeight={info.height}\nimExt=.jpg\n"
                f"; frames {a}-{b} of {src.name}\n"
            )
            out[split].append(read_seqinfo(dst))
    return out


def list_split(root: Path, split: str, sequences: tuple[str, ...] | list[str] | None = None) -> list[SeqInfo]:
    base = root / split
    if not base.exists():
        raise FileNotFoundError(f"{base} not found; run scripts/prepare_mot17.py")
    names = sequences or SEQUENCES
    return [read_seqinfo(base / s) for s in names]
