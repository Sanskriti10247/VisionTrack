import json

import pytest

from visiontrack.config import PipelineConfig, load_config
from visiontrack.pipeline import VisionPipeline


def test_end_to_end_outputs(tiny_video, scripted_detector, tmp_path):
    cfg = PipelineConfig.model_validate(
        {
            "analytics": {"lines": [{"name": "mid", "p1": [80, 0], "p2": [80, 120]}]},
            "output": {"save_video": True},
        }
    )
    res = VisionPipeline(cfg, detector=scripted_detector).process_video(tiny_video, tmp_path / "out")
    s = res.summary
    assert s["performance"]["frames_processed"] == 30
    assert s["analytics"]["unique_objects"]["total"] == 1
    # The square moves left to right across x=80 exactly once before wrapping around.
    assert s["analytics"]["lines"][0]["raw_crossing_events"] >= 1
    for key in ("summary", "events", "tracks", "trajectories", "frames", "mot", "annotated_video"):
        assert key in res.files

    rows = (tmp_path / "out" / "mot_results.txt").read_text().strip().splitlines()
    first = rows[0].split(",")
    assert len(first) == 10 and first[0] == "1" and first[1] == "1"
    frames = [json.loads(line) for line in (tmp_path / "out" / "frames.jsonl").read_text().splitlines()]
    assert len(frames) == 30 and frames[3]["tracks"][0]["track_id"] == 1
    lat = s["performance"]["latency_ms"]
    for k in ("detector_ms", "tracker_ms", "analytics_ms", "end_to_end_ms"):
        assert lat[k]["n"] == 30 and lat[k]["p95"] >= lat[k]["p50"]


def test_batched_detection_matches_unbatched(tiny_video, scripted_detector, tmp_path):
    single = VisionPipeline(PipelineConfig(), scripted_detector).process_video(tiny_video, tmp_path / "a")
    cfg = PipelineConfig.model_validate({"detector": {"batch_size": 4, "conf_threshold": 0.1}})
    scripted_detector.cfg = cfg.detector
    batched = VisionPipeline(cfg, scripted_detector).process_video(tiny_video, tmp_path / "b")
    assert (tmp_path / "a" / "mot_results.txt").read_text() == (
        tmp_path / "b" / "mot_results.txt"
    ).read_text()
    assert (
        single.summary["performance"]["frames_processed"]
        == batched.summary["performance"]["frames_processed"]
    )


def test_stride_halves_frames(tiny_video, scripted_detector):
    cfg = PipelineConfig.model_validate({"video": {"stride": 2}, "output": {"save_video": False}})
    res = VisionPipeline(cfg, scripted_detector).process_video(tiny_video, None)
    assert res.summary["performance"]["frames_processed"] == 15


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        PipelineConfig.model_validate({"detector": {"imgsz": 650}})
    with pytest.raises(ValueError):
        PipelineConfig.model_validate({"tracker": {"track_low_thresh": 0.9, "track_high_thresh": 0.5}})
    with pytest.raises(ValueError):
        PipelineConfig.model_validate({"detector": {"unknown_key": 1}})
    p = tmp_path / "c.yaml"
    p.write_text("detector:\n  imgsz: 960\n")
    cfg = load_config(p, {"tracker": {"track_buffer": 60}})
    assert cfg.detector.imgsz == 960 and cfg.tracker.track_buffer == 60
