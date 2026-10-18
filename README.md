# VisionTrack

Real-time object detection, multi-object tracking and track-based video analytics.

Work in progress — see [docs/PLAN.md](docs/PLAN.md) for the milestones.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[onnx,api,eval,dev]"
visiontrack run path/to/video.mp4 -o outputs/demo
```
