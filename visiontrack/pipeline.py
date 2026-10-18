"""End-to-end video pipeline: ingest -> detect -> track -> analyse -> render/serialise.

Each stage is a separate object (Detector, Tracker, AnalyticsEngine) wired
together here; nothing in this module knows which backend is in use.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from visiontrack import __version__
from visiontrack.analytics.engine import AnalyticsEngine
from visiontrack.config import PipelineConfig
from visiontrack.detection import Detector, build_detector
from visiontrack.io.video import Frame, VideoSource, VideoWriter
from visiontrack.timing import LatencyRecorder, peak_rss_mb
from visiontrack.tracking import build_tracker
from visiontrack.types import Detections, FrameResult
from visiontrack.visualize import draw_frame

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int | None], None]


class Cancelled(RuntimeError):
    pass


@dataclass
class RunResult:
    summary: dict
    output_dir: Path | None
    files: dict[str, str] = field(default_factory=dict)


class VisionPipeline:
    def __init__(self, cfg: PipelineConfig, detector: Detector | None = None) -> None:
        self.cfg = cfg
        self.detector = detector if detector is not None else build_detector(cfg.detector)

    # ------------------------------------------------------------------ video
    def process_video(
        self,
        source: str | Path | int,
        output_dir: str | Path | None = None,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
        warmup: bool = True,
    ) -> RunResult:
        cfg = self.cfg
        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        src = VideoSource(source, cfg.video.stride, cfg.video.start_frame, cfg.video.max_frames)
        fps = src.output_fps
        tracker = build_tracker(cfg.tracker, frame_rate=fps)
        engine = AnalyticsEngine(cfg.analytics, fps, (src.width, src.height), self.detector.names)
        lat = LatencyRecorder()
        warmup_ms = self.detector.warmup(shape=(src.height, src.width)) if warmup else 0.0

        writer = None
        if out_dir and cfg.output.save_video:
            writer = VideoWriter(out_dir / "annotated.mp4", fps, (src.width, src.height))
        mot_rows: list[str] = []
        frame_records: list[dict] = []
        decode_ms: dict[int, float] = {}
        expected = src.expected_frames
        done = 0
        t_start = time.perf_counter()

        try:
            batch: list[Frame] = []
            t_read = time.perf_counter()
            for frame in src:
                decode_ms[frame.index] = (time.perf_counter() - t_read) * 1000
                batch.append(frame)
                if len(batch) < cfg.detector.batch_size:
                    t_read = time.perf_counter()
                    continue
                for result in self._process_batch(batch, tracker, engine, lat):
                    self._emit(result, engine, writer, batch, mot_rows, frame_records, decode_ms, lat)
                    done += 1
                batch = []
                if progress:
                    progress(done, expected)
                if cancel is not None and cancel.is_set():
                    raise Cancelled("job cancelled")
                t_read = time.perf_counter()
            if batch:
                for result in self._process_batch(batch, tracker, engine, lat):
                    self._emit(result, engine, writer, batch, mot_rows, frame_records, decode_ms, lat)
                    done += 1
                if progress:
                    progress(done, expected)
        finally:
            src.release()
            video_path = writer.close() if writer else None

        engine.finalize()
        wall_s = time.perf_counter() - t_start
        summary = {
            "visiontrack_version": __version__,
            "video": src.info(),
            "config": cfg.model_dump(mode="json"),
            "detector": self.detector.info(),
            "analytics": engine.summary(),
            "performance": {
                "scope": "single video run; latencies are per processed frame in milliseconds",
                "frames_processed": done,
                "wall_time_s": round(wall_s, 3),
                "throughput_fps": round(done / wall_s, 2) if wall_s > 0 else None,
                "warmup_ms": round(warmup_ms, 1),
                "latency_ms": lat.summary(),
                "peak_rss_mb": round(peak_rss_mb(), 1),
            },
        }

        files: dict[str, str] = {}
        if out_dir:
            files = self._write_outputs(out_dir, summary, engine, mot_rows, frame_records)
            if video_path:
                files["annotated_video"] = str(video_path)
        return RunResult(summary, out_dir, files)

    def _process_batch(self, batch: list[Frame], tracker, engine: AnalyticsEngine, lat: LatencyRecorder):
        t0 = time.perf_counter()
        dets_list = self.detector.detect_batch([f.image for f in batch])
        det_ms = (time.perf_counter() - t0) * 1000 / len(batch)
        det_breakdown = dict(self.detector.last_timings)
        for frame, dets in zip(batch, dets_list, strict=True):
            timings = {"detector_ms": det_ms}
            for k in ("preprocess_ms", "inference_ms", "postprocess_ms"):
                if k in det_breakdown:
                    timings[f"detector_{k}"] = det_breakdown[k]
            t1 = time.perf_counter()
            tracks = tracker.update(dets)
            timings["tracker_ms"] = (time.perf_counter() - t1) * 1000
            t2 = time.perf_counter()
            engine.update(frame.index, frame.timestamp_s, tracks, tracker.ended_track_ids)
            timings["analytics_ms"] = (time.perf_counter() - t2) * 1000
            lat.add_many(timings)
            yield FrameResult(frame.index, frame.timestamp_s, dets, tracks, timings)

    def _emit(
        self, result: FrameResult, engine, writer, batch, mot_rows, frame_records, decode_ms, lat
    ) -> None:
        cfg = self.cfg
        t0 = time.perf_counter()
        frame_no = result.frame_index + 1  # MOTChallenge frames are 1-based
        for t in result.tracks:
            x1, y1, x2, y2 = (float(v) for v in t.xyxy)
            mot_rows.append(
                f"{frame_no},{t.track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{t.score:.4f},-1,-1,-1"
            )
        frame_records.append(
            {
                "frame_index": result.frame_index,
                "timestamp_s": round(result.timestamp_s, 3),
                "num_detections": len(result.detections),
                "tracks": [t.to_record(self.detector.names) for t in result.tracks],
            }
        )
        if writer is not None:
            image = next(f.image for f in batch if f.index == result.frame_index)
            n = cfg.output.trail_length
            trail_map = (
                {t.track_id: engine.recent_trail(t.track_id, n) for t in result.tracks}
                if cfg.output.draw_trajectories and n > 1
                else None
            )
            live = engine.live_counts()
            lines = [(ln.cfg.name, ln.p1, ln.p2, live["lines"].get(ln.cfg.name, {})) for ln in engine.lines]
            zones = [(z.cfg.name, z.polygon, live["zone_occupancy"].get(z.cfg.name, 0)) for z in engine.zones]
            frame_ms = sum(
                v for k, v in result.timings_ms.items() if k in ("detector_ms", "tracker_ms", "analytics_ms")
            )
            hud = [
                f"frame {result.frame_index}  t={result.timestamp_s:.2f}s  {frame_ms:.0f} ms/frame",
                f"tracks {len(result.tracks)}  unique {live['unique_objects']}",
            ]
            if cfg.output.draw_low_score_detections:
                from visiontrack.visualize import draw_detections

                draw_detections(image, result.detections, self.detector.names)
            draw_frame(image, result.tracks, self.detector.names, trail_map, lines, zones, hud)
            writer.write(image)
        render_ms = (time.perf_counter() - t0) * 1000
        decode = decode_ms.pop(result.frame_index, 0.0)
        lat.add("render_write_ms", render_ms)
        lat.add("decode_ms", decode)
        # Complete per-frame cost of the pipeline, as opposed to detector-only latency.
        lat.add(
            "end_to_end_ms",
            decode
            + sum(result.timings_ms[k] for k in ("detector_ms", "tracker_ms", "analytics_ms"))
            + render_ms,
        )

    @staticmethod
    def _write_outputs(
        out_dir: Path, summary, engine: AnalyticsEngine, mot_rows, frame_records
    ) -> dict[str, str]:
        files = {
            "summary": out_dir / "summary.json",
            "events": out_dir / "events.json",
            "tracks": out_dir / "tracks.json",
            "trajectories": out_dir / "trajectories.json",
            "frames": out_dir / "frames.jsonl",
            "mot": out_dir / "mot_results.txt",
        }
        files["summary"].write_text(json.dumps(summary, indent=2))
        files["events"].write_text(json.dumps(engine.events, indent=1))
        files["tracks"].write_text(json.dumps(engine.track_table(), indent=1))
        files["trajectories"].write_text(json.dumps({str(k): v for k, v in engine.trajectories().items()}))
        with open(files["frames"], "w") as f:
            for rec in frame_records:
                f.write(json.dumps(rec) + "\n")
        files["mot"].write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""))
        return {k: str(v) for k, v in files.items()}

    # ------------------------------------------------------------------ image
    def detect_image(self, image: np.ndarray) -> tuple[Detections, dict]:
        t0 = time.perf_counter()
        dets = self.detector.detect(image)
        timings = {**self.detector.last_timings, "total_ms": (time.perf_counter() - t0) * 1000}
        return dets, timings
