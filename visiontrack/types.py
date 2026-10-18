"""Core data containers shared by the detector, tracker and analytics stages.

Everything is kept as plain numpy arrays so that stages can be swapped
(PyTorch vs ONNX detector, ByteTrack vs another tracker) without touching
the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO class names for the 80-class detectors shipped by ultralytics.
COCO_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# Default classes for street scenes: people and road vehicles.
DEFAULT_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 5, 7)


@dataclass
class Detections:
    """Detections for a single frame, in original-frame pixel coordinates."""

    xyxy: np.ndarray  # (N, 4) float32
    confidence: np.ndarray  # (N,) float32
    class_id: np.ndarray  # (N,) int64

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        self.class_id = np.asarray(self.class_id, dtype=np.int64).reshape(-1)
        n = len(self.xyxy)
        if len(self.confidence) != n or len(self.class_id) != n:
            raise ValueError(
                f"inconsistent detection arrays: boxes={n}, conf={len(self.confidence)}, cls={len(self.class_id)}"
            )

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.xyxy)

    def filter(self, mask: np.ndarray) -> Detections:
        return Detections(self.xyxy[mask], self.confidence[mask], self.class_id[mask])

    def to_records(self, names: tuple[str, ...] | list[str] = COCO_NAMES) -> list[dict]:
        out = []
        for box, conf, cls in zip(self.xyxy, self.confidence, self.class_id, strict=True):
            c = int(cls)
            out.append(
                {
                    "bbox_xyxy": [round(float(v), 2) for v in box],
                    "confidence": round(float(conf), 4),
                    "class_id": c,
                    "class_name": names[c] if 0 <= c < len(names) else str(c),
                }
            )
        return out


@dataclass
class TrackedObject:
    """A confirmed track's state as reported for one frame."""

    track_id: int
    xyxy: np.ndarray  # (4,)
    score: float  # confidence of the detection that updated the track this frame
    class_id: int  # majority-vote class over the track's history
    age: int  # frames since the track was started
    hits: int  # number of frames with an associated detection
    time_since_update: int = 0

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre point: where an object touches the ground plane in most street footage."""
        x1, _, x2, y2 = self.xyxy
        return float((x1 + x2) / 2.0), float(y2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    def to_record(self, names: tuple[str, ...] | list[str] = COCO_NAMES) -> dict:
        return {
            "track_id": int(self.track_id),
            "bbox_xyxy": [round(float(v), 2) for v in self.xyxy],
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "class_name": names[self.class_id] if 0 <= self.class_id < len(names) else str(self.class_id),
            "age": int(self.age),
            "hits": int(self.hits),
        }


@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame."""

    frame_index: int  # index in the source video (not the sampled index)
    timestamp_s: float
    detections: Detections
    tracks: list[TrackedObject] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
