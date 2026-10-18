# VisionTrack – build plan

## Environment the plan was made for

| | |
|---|---|
| CPU | Intel Core i5-13420H, 12 threads |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB (shared with other workloads on this machine) |
| RAM | 16 GB (often only ~3 GB free) |
| Python | 3.12, PyTorch 2.6 + CUDA 12.4, ultralytics 8.3 |

Consequences: everything must run on CPU, GPU paths are optional, and benchmark numbers
need to be read with the shared-machine caveat in mind (recorded with each result).

## Datasets

* **Tracking (with ground truth):** MOT17 train split, 7 sequences, 5,316 frames, pedestrians.
  MOT17 test GT is not public, so the train sequences are split per sequence into
  **half-1 (tuning)** and **half-2 (held-out evaluation)**, following the protocol used by ByteTrack
  and CenterTrack. All parameter sweeps run on half-1; headline numbers are reported on half-2.
* **Detection:** COCO val2017, restricted to person + road-vehicle classes, plus MOT17 half-2
  person boxes (in-domain check). COCO detection labels are *never* used for tracking metrics.

## Milestones

| # | Milestone | Acceptance criteria |
|---|-----------|---------------------|
| M1 | Minimal pipeline: video in → YOLO → ByteTrack → annotated video + basic stats | runs on CPU on a MOT17 sequence; persistent IDs visible; unit tests for video IO, track lifecycle |
| M2 | Trajectory analytics | line crossings with hysteresis and per-track de-duplication, zone enter/exit with dwell, per-class unique counts; tests for every duplicate-counting edge case |
| M3 | Evaluation framework | detection P/R/mAP via pycocotools; HOTA/IDF1/MOTA/IDSW/Frag via TrackEval on MOT17 GT; detections cached so tracker experiments reuse identical inputs |
| M4 | Controlled experiments + robustness | sweeps over model size, input resolution, detector threshold, tracker settings (tuned on half-1, reported on half-2); synthetic blur/low-light/downscale perturbations; recall by visibility and object size; failure cases saved and categorised |
| M5 | ONNX export + edge inference | ONNX Runtime backend with its own pre/post-processing; raw-output and mAP parity vs PyTorch; FP32/FP16/INT8, resolution and batch sweeps; p50/p95, warm-up, model size, peak memory; detector-only vs end-to-end latency |
| M6 | FastAPI service | detect/track/jobs/health/metrics endpoints; background jobs; upload validation, size limits, cleanup; model cache; API tests |
| M7 | Dashboard | upload, status, annotated playback, counts, trajectories, events, latency stats, config comparison; metric definitions on screen |
| M8 | Packaging + report | Dockerfile (CPU) + GPU variant, architecture diagram, setup docs, API docs, technical report with real numbers, demo media |

## Rules for results

* Every experiment writes a JSON record with model, dataset split, hardware, runtime settings and metrics.
* No number goes in the report unless a script in this repo produced it.
* Tracking metrics only come from data with tracking ground truth (MOT17).
* Benchmarks compare configurations under the same conditions (same frames, same machine, warm-up excluded, repeated runs).
