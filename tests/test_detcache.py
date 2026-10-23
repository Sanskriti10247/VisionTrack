import cv2
import numpy as np

from visiontrack.config import DetectorConfig
from visiontrack.eval.detcache import build_cache, load_cache
from visiontrack.eval.mot17 import read_seqinfo
from visiontrack.types import Detections


class RampDetector:
    """Returns frame-dependent detections with a spread of scores."""

    def __init__(self):
        self.calls = 0
        self.names = ("person",)
        self.last_timings = {}

    def warmup(self, *a, **k):
        return 0.0

    def info(self):
        return {"backend": "fake"}

    def detect_batch(self, images):
        out = []
        for img in images:
            k = int(img[0, 0, 0])
            boxes = [[k, k, k + 10, k + 20], [50, 50, 60, 70]]
            out.append(Detections(boxes, [0.9, 0.05], [0, 2]))
            self.calls += 1
        return out


def test_cache_roundtrip_and_filters(tmp_path):
    seq = tmp_path / "SEQ"
    (seq / "img1").mkdir(parents=True)
    for i in range(1, 8):
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.png"), np.full((32, 32, 3), i, np.uint8))
    (seq / "seqinfo.ini").write_text("[Sequence]\nseqLength=7\nframeRate=30\nimWidth=32\nimHeight=32\n")
    info = read_seqinfo(seq)
    det = RampDetector()
    out = build_cache(DetectorConfig(), [info], tmp_path / "cache", batch_size=3, detector=det)
    assert det.calls == 7
    frames = load_cache(out, "SEQ")
    assert len(frames) == 7 and len(frames[0]) == 2
    np.testing.assert_allclose(frames[4].xyxy[0], [5, 5, 15, 25])
    assert all(len(f) == 1 for f in load_cache(out, "SEQ", conf=0.1))
    assert all(f.class_id.tolist() == [2] for f in load_cache(out, "SEQ", classes=[2]))
    # second call is a no-op because the cache exists
    build_cache(DetectorConfig(), [info], tmp_path / "cache", detector=det)
    assert det.calls == 7
