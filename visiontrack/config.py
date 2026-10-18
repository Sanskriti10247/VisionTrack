"""Typed configuration for every pipeline stage.

Configs are pydantic models so they validate user input (YAML files, API
requests) in one place and serialise cleanly into experiment records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visiontrack.types import DEFAULT_CLASSES

Point = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_Strict):
    backend: Literal["ultralytics", "onnx"] = "ultralytics"
    weights: str = "yolo11n.pt"
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.7, gt=0.0, le=1.0, description="NMS IoU threshold")
    imgsz: int = Field(640, ge=32, le=4096)
    classes: list[int] | None = Field(default_factory=lambda: list(DEFAULT_CLASSES))
    device: str = Field("auto", description="'auto', 'cpu', 'cuda' or 'cuda:N'")
    half: bool = Field(False, description="FP16 inference (GPU only)")
    max_det: int = Field(300, ge=1)
    batch_size: int = Field(1, ge=1, le=64)

    @field_validator("imgsz")
    @classmethod
    def _stride_multiple(cls, v: int) -> int:
        if v % 32:
            raise ValueError("imgsz must be a multiple of 32 (YOLO stride)")
        return v


class TrackerConfig(_Strict):
    """ByteTrack parameters. Names follow the official implementation where possible."""

    type: Literal["bytetrack", "ultralytics_bytetrack"] = "bytetrack"
    track_high_thresh: float = Field(0.5, ge=0.0, le=1.0, description="first-association score threshold")
    track_low_thresh: float = Field(0.1, ge=0.0, le=1.0, description="minimum score for second association")
    new_track_thresh: float = Field(0.6, ge=0.0, le=1.0, description="minimum score to start a new track")
    match_thresh: float = Field(0.8, gt=0.0, le=1.0, description="max (1-IoU) cost, first association")
    second_match_thresh: float = Field(
        0.5, gt=0.0, le=1.0, description="max (1-IoU) cost, low-score association"
    )
    unconfirmed_match_thresh: float = Field(0.7, gt=0.0, le=1.0)
    track_buffer: int = Field(30, ge=0, description="frames a lost track is kept (at 30 FPS)")
    fuse_score: bool = True
    use_low_score_association: bool = Field(
        True, description="False disables ByteTrack's second stage (high-score-only, SORT-like ablation)"
    )
    class_aware: bool = Field(True, description="forbid associating detections of different classes")
    min_box_area: float = Field(0.0, ge=0.0, description="drop output tracks smaller than this (px^2)")

    @model_validator(mode="after")
    def _ordering(self) -> TrackerConfig:
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh must be <= track_high_thresh")
        return self


class LineConfig(_Strict):
    name: str
    p1: Point
    p2: Point
    hysteresis_px: float = Field(
        4.0, ge=0.0, description="anchor must move this far past the line before a crossing registers"
    )


class ZoneConfig(_Strict):
    name: str
    polygon: list[Point] = Field(min_length=3)
    min_frames_inside: int = Field(
        3, ge=1, description="consecutive frames inside before an entry is confirmed"
    )
    min_frames_outside: int = Field(
        3, ge=1, description="consecutive frames outside before an exit is confirmed"
    )


class AnalyticsConfig(_Strict):
    lines: list[LineConfig] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)
    normalized_coordinates: bool = Field(
        False, description="line/zone coordinates are fractions of frame width/height"
    )
    anchor: Literal["bottom_center", "center"] = "bottom_center"
    min_track_hits: int = Field(
        3, ge=1, description="a track must be matched in this many frames to count as a unique object"
    )
    count_once_per_track: bool = Field(
        True, description="each track contributes at most one count per line and direction"
    )
    trajectory_max_points: int = Field(2000, ge=2)


class VideoConfig(_Strict):
    stride: int = Field(1, ge=1, description="process every Nth frame")
    start_frame: int = Field(0, ge=0)
    max_frames: int | None = Field(None, ge=1, description="maximum number of processed frames")


class OutputConfig(_Strict):
    save_video: bool = True
    draw_trajectories: bool = True
    trail_length: int = Field(40, ge=0)
    draw_low_score_detections: bool = False


class PipelineConfig(_Strict):
    # ByteTrack's second association stage needs low-score boxes, so the
    # detector threshold must not be higher than track_low_thresh.
    detector: DetectorConfig = Field(default_factory=lambda: DetectorConfig(conf_threshold=0.1))
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _detector_feeds_tracker(self) -> PipelineConfig:
        if (
            self.detector.conf_threshold > self.tracker.track_low_thresh
            and self.tracker.use_low_score_association
        ):
            # Not an error (it is a valid ablation) but it silently disables
            # most of ByteTrack's benefit, so make the effective value explicit.
            self.tracker = self.tracker.model_copy(
                update={"track_low_thresh": min(self.detector.conf_threshold, self.tracker.track_high_thresh)}
            )
        return self


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> PipelineConfig:
    data: dict = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return PipelineConfig.model_validate(data)


def _deep_merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
