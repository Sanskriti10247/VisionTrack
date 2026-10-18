"""Command-line entry point: ``visiontrack <command> ...``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="YAML pipeline config")
    p.add_argument("--weights", help="detector weights (.pt or .onnx)")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--conf", type=float, help="detector confidence threshold")
    p.add_argument("--iou", type=float, help="NMS IoU threshold")
    p.add_argument("--device", help="auto | cpu | cuda:0")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU)")
    p.add_argument("--batch", type=int, help="detector batch size")


def _overrides(args: argparse.Namespace) -> dict:
    det: dict = {}
    if args.weights:
        det["weights"] = args.weights
        det["backend"] = "onnx" if args.weights.endswith(".onnx") else "ultralytics"
    for key, attr in (
        ("imgsz", "imgsz"),
        ("conf_threshold", "conf"),
        ("iou_threshold", "iou"),
        ("device", "device"),
        ("batch_size", "batch"),
    ):
        if getattr(args, attr, None) is not None:
            det[key] = getattr(args, attr)
    if getattr(args, "half", False):
        det["half"] = True
    return {"detector": det} if det else {}


def cmd_run(args: argparse.Namespace) -> int:
    from visiontrack.config import load_config
    from visiontrack.pipeline import VisionPipeline

    overrides = _overrides(args)
    video = {}
    if args.stride:
        video["stride"] = args.stride
    if args.max_frames:
        video["max_frames"] = args.max_frames
    if video:
        overrides["video"] = video
    if args.tracker:
        overrides.setdefault("tracker", {})["type"] = args.tracker
    if args.no_video:
        overrides["output"] = {"save_video": False}
    cfg = load_config(args.config, overrides)
    pipe = VisionPipeline(cfg)

    def progress(done: int, total: int | None) -> None:
        if done % 50 == 0 or (total and done == total):
            print(f"\r{done}/{total or '?'} frames", end="", file=sys.stderr, flush=True)

    result = pipe.process_video(args.source, args.output, progress=progress)
    print(file=sys.stderr)
    s = result.summary
    perf = s["performance"]
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "frames": perf["frames_processed"],
                "throughput_fps": perf["throughput_fps"],
                "end_to_end_ms_p50": perf["latency_ms"].get("end_to_end_ms", {}).get("p50"),
                "detector_ms_p50": perf["latency_ms"].get("detector_ms", {}).get("p50"),
                "unique_objects": s["analytics"]["unique_objects"],
                "lines": {ln["name"]: ln["counts"] for ln in s["analytics"]["lines"]},
            },
            indent=2,
        )
    )
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    import cv2

    from visiontrack.config import load_config
    from visiontrack.detection import build_detector

    cfg = load_config(args.config, _overrides(args))
    det = build_detector(cfg.detector)
    images = [Path(p) for p in args.images]
    out = []
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"cannot read {p}", file=sys.stderr)
            return 2
        d = det.detect(img)
        out.append({"image": str(p), "detections": d.to_records(det.names), "timings_ms": det.last_timings})
    print(json.dumps(out, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "visiontrack.api.app:create_app", factory=True, host=args.host, port=args.port, log_level="info"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="visiontrack")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="detect + track + analyse a video, image folder or camera")
    run.add_argument("source", help="video file, image directory, camera index or stream URL")
    run.add_argument("-o", "--output", type=Path, default=Path("outputs/run"))
    run.add_argument("--stride", type=int)
    run.add_argument("--max-frames", type=int)
    run.add_argument("--tracker", choices=["bytetrack", "ultralytics_bytetrack"])
    run.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    _add_common(run)
    run.set_defaults(func=cmd_run)

    det = sub.add_parser("detect", help="run the detector on images and print JSON")
    det.add_argument("images", nargs="+")
    _add_common(det)
    det.set_defaults(func=cmd_detect)

    srv = sub.add_parser("serve", help="start the FastAPI service")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
