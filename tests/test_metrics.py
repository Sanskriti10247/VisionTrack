"""Sanity checks of the metric wrappers on hand-made data with known answers."""

from pathlib import Path

import numpy as np
import pytest

from visiontrack.eval.detection import coco_evaluate, mot17_gt_to_coco
from visiontrack.eval.mot17 import read_seqinfo
from visiontrack.eval.tracking import trackeval_mot17, write_results

pytest.importorskip("trackeval")
pytest.importorskip("pycocotools")

N_FRAMES = 20


def _make_seq(root: Path, name: str = "SEQ-01") -> Path:
    """Two pedestrians walking towards each other plus one distractor (static person)."""
    seq = root / "gt" / name
    (seq / "gt").mkdir(parents=True)
    rows = []
    for f in range(1, N_FRAMES + 1):
        rows.append([f, 1, 10 + 10 * f, 50, 40, 100, 1, 1, 1.0])
        rows.append([f, 2, 400 - 10 * f, 300, 40, 100, 1, 1, 1.0])
        rows.append([f, 3, 600, 50, 40, 100, 0, 7, 1.0])  # static person: distractor
    np.savetxt(seq / "gt" / "gt.txt", np.array(rows), delimiter=",", fmt="%g")
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={N_FRAMES}\nimWidth=800\nimHeight=600\nimExt=.jpg\n"
    )
    return seq


def _rows(swap_at: int | None = None, drop_id: int | None = None, distractor: bool = False) -> list[str]:
    out = []
    for f in range(1, N_FRAMES + 1):
        a, b = 1, 2
        if swap_at is not None and f >= swap_at:
            a, b = 2, 1
        if drop_id != 1:
            out.append(f"{f},{a},{10 + 10 * f},50,40,100,0.9,-1,-1,-1")
        if drop_id != 2:
            out.append(f"{f},{b},{400 - 10 * f},300,40,100,0.9,-1,-1,-1")
        if distractor:
            out.append(f"{f},9,600,50,40,100,0.9,-1,-1,-1")
    return out


def _eval(tmp_path: Path, rows: list[str]) -> dict:
    seq = _make_seq(tmp_path)
    write_results(rows, tmp_path / "trackers" / "t" / "data" / "SEQ-01.txt")
    return trackeval_mot17(tmp_path / "gt", tmp_path / "trackers", "t", [read_seqinfo(seq)])["combined"]


def test_perfect_tracking(tmp_path):
    m = _eval(tmp_path, _rows())
    assert m["HOTA"] == pytest.approx(100.0)
    assert m["MOTA"] == pytest.approx(100.0) and m["IDF1"] == pytest.approx(100.0)
    assert m["IDSW"] == 0 and m["FP"] == 0 and m["FN"] == 0 and m["GT_IDs"] == 2


def test_distractor_matches_are_not_false_positives(tmp_path):
    m = _eval(tmp_path, _rows(distractor=True))
    assert m["FP"] == 0 and m["MOTA"] == pytest.approx(100.0)


def test_identity_swap_is_counted(tmp_path):
    m = _eval(tmp_path, _rows(swap_at=11))
    assert m["IDSW"] == 2
    assert m["MOTA"] == pytest.approx(100 * (1 - 2 / 40))
    assert m["IDF1"] == pytest.approx(50.0)
    assert m["HOTA"] < 100


def test_missed_object(tmp_path):
    m = _eval(tmp_path, _rows(drop_id=2))
    assert m["FN"] == N_FRAMES and m["MOTA"] == pytest.approx(50.0)
    assert m["ML"] == 1 and m["MT"] == 1


def test_coco_wrapper_perfect_and_operating_point():
    gt = {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 100], "area": 5000, "iscrowd": 0},
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [100, 50, 60, 120],
                "area": 7200,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    perfect = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 100], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [100, 50, 60, 120], "score": 0.8},
    ]
    r = coco_evaluate(gt, perfect, [1])
    assert r["mAP50_95"] == pytest.approx(1.0) and r["mAP50"] == pytest.approx(1.0)
    op = r["operating_point"]
    assert op["precision"] == 1.0 and op["recall"] == 1.0

    with_fp = perfect + [{"image_id": 1, "category_id": 1, "bbox": [300, 300, 50, 50], "score": 0.5}]
    r = coco_evaluate(gt, with_fp, [1], operating_conf=0.3)
    assert r["operating_point"]["fp"] == 1 and r["operating_point"]["precision"] == pytest.approx(
        2 / 3, abs=1e-3
    )
    # The FP scores below both TPs, so AP is unaffected while precision@0.3 drops.
    assert r["mAP50"] == pytest.approx(1.0)
    r = coco_evaluate(gt, with_fp, [1], operating_conf=0.85)
    assert r["operating_point"]["precision"] == 1.0 and r["operating_point"]["recall"] == 0.5


def test_mot17_gt_conversion_marks_ignore_regions(tmp_path):
    seq = _make_seq(tmp_path)
    gt_txt = seq / "gt" / "gt.txt"
    rows = np.loadtxt(gt_txt, delimiter=",")
    rows = np.vstack([rows, [1, 4, 200, 200, 40, 100, 1, 1, 0.1]])  # barely visible pedestrian
    np.savetxt(gt_txt, rows, delimiter=",", fmt="%g")
    coco = mot17_gt_to_coco(gt_txt, N_FRAMES)
    crowd = [a for a in coco["annotations"] if a["iscrowd"]]
    assert len(coco["images"]) == N_FRAMES
    assert len(crowd) == N_FRAMES + 1  # 20 static-person boxes + 1 low-visibility pedestrian
