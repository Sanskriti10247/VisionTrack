# Deployment

How to build and run VisionTrack in a container on a machine you control. Only the CLI
(`run`, `detect`) is containerised, on CPU. There is no GPU image yet, and no service to expose,
since the HTTP API isn't written (see the [roadmap](../README.md#roadmap)). When it is, the
image will also need the `api` extra and `serve --host 0.0.0.0`, because `serve` binds to
127.0.0.1 by default.

## Image

`docker/Dockerfile` is a two-stage build on `python:3.12.15-slim-trixie`, pinned by digest
(override with `--build-arg PYTHON_IMAGE=...`).

The builder stage creates a venv at `/opt/venv`, installs CPU-only `torch==2.6.0` /
`torchvision==0.21.0` from the PyTorch CPU index first so `ultralytics` doesn't pull the CUDA
wheels, then installs `.[onnx]`. Torch versions can be changed with
`--build-arg TORCH_VERSION=... TORCHVISION_VERSION=...`.

The runtime stage adds `libgl1` and `libglib2.0-0t64` for OpenCV, copies the venv and `configs/`,
and runs as the non-root user `visiontrack` (uid/gid 10001) with `ENTRYPOINT ["visiontrack"]`.
The healthcheck only imports `visiontrack.pipeline`, `onnxruntime` and `cv2`; the CLI has no
long-running process to probe.

`ffmpeg` is not installed, so `annotated.mp4` stays MPEG-4 Part 2 (`mp4v`). It plays in VLC and
ffplay but not in every browser. Adding `ffmpeg` to the runtime `apt-get` line gives H.264 output
for roughly 100 to 200 MB more.

`.dockerignore` limits the build context to `pyproject.toml`, `README.md`, `visiontrack/` and
`configs/`.

Built with Docker 29.7.2 / Compose 5.5.1, the image is 2.7 GB on disk (616 MB compressed) and
reports `torch 2.6.0+cpu` and `onnxruntime 1.31.0`.

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile -t visiontrack:cpu .
# or
docker compose build
```

## Models

Weights aren't baked into the image: they're large and change independently of the code.

The default is to bind-mount `./models` read-only at `/app/models`. The working directory is
`/app` and the detector looks up bare names like `yolo11n.pt` in `./models/`, so both
`--weights yolo11n.pt` and `--weights models/onnx/yolo11n_640_fp32.onnx` resolve in the
container. To populate it on the host:

```bash
python -c "from ultralytics.utils.downloads import attempt_download_asset as d; d('models/yolo11n.pt')"
python scripts/export_models.py --no-gpu   # optional: ONNX variants into models/onnx/
```

For a self-contained image, bake a model into a derived image. Build it from a context that
contains the file, since the repo's `.dockerignore` excludes `models/`:

```dockerfile
FROM visiontrack:cpu
COPY --chown=visiontrack:visiontrack models/onnx/yolo11n_640_fp32.onnx /app/models/onnx/
```

`YOLO_OFFLINE=true` plus the read-only mount means a missing `.pt` fails immediately instead of
being downloaded at runtime. `.onnx` weights are the better choice in a container: the ONNX
Runtime backend never loads PyTorch at inference time.

## Run

With compose (inputs from `./data`, results to `./outputs`):

```bash
docker compose run --rm visiontrack run /app/data/MOT17/train/MOT17-04-FRCNN/img1 \
    --config configs/mot17-04-demo.yaml \
    --weights models/onnx/yolo11n_640_fp32.onnx --imgsz 640 --device cpu \
    --max-frames 150 -o /app/outputs/docker_demo

docker compose run --rm visiontrack detect /app/data/samples/bus.jpg \
    --weights models/onnx/yolo11n_640_fp32.onnx --device cpu
```

The `detect` call returns the same 5 detections as on the host (bus 0.94, persons 0.90 / 0.85 /
0.83 / 0.40). A 60-frame `run` of the same clip writes all 7 output files, owned by the host
user, at about 7 FPS.

With plain `docker run`:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" -v "$PWD/outputs:/app/outputs" \
    --cpus 4 --memory 3g \
    visiontrack:cpu run /app/data/my_video.mp4 --weights models/onnx/yolo11n_640_fp32.onnx \
    --imgsz 640 --device cpu -o /app/outputs/my_video
```

Put your own videos under `./data` and pass the `/app/data/...` path. Outputs must go under
`/app/outputs`, because the compose service has a read-only root filesystem.

## Volumes

| Container path | Host (compose) | Mode | Purpose |
|---|---|---|---|
| `/app/models` | `./models` | ro | `.pt` / `.onnx` weights |
| `/app/data` | `./data` | ro | input videos, image folders, sample images |
| `/app/outputs` | `./outputs` | rw | `annotated.mp4`, `summary.json`, `events.json`, `tracks.json`, `trajectories.json`, `frames.jsonl`, `mot_results.txt` |
| `/tmp` | tmpfs, 256 MB | rw | ultralytics settings (`YOLO_CONFIG_DIR`) and scratch |
| `/app/runs` | tmpfs, 64 MB | rw | see below |

The ultralytics predictor creates `runs/detect/predict` under the working directory even though
VisionTrack saves nothing there. On a read-only root filesystem the `.pt` path fails with
`OSError: Read-only file system: 'runs'` unless `/app/runs` is writable.

## Environment variables

Set in the image and compose:

| Variable | Default | Meaning |
|---|---|---|
| `OMP_NUM_THREADS` | `4` | intra-op threads for PyTorch / OpenCV (compose sets it from `VT_THREADS`) |
| `YOLO_CONFIG_DIR` | `/tmp` | writable ultralytics settings dir for the non-root user |
| `YOLO_OFFLINE` | `true` | no ultralytics network calls (downloads, update checks) |

Compose only (shell or a `.env` file next to `docker-compose.yml`):

| Variable | Default | Meaning |
|---|---|---|
| `VT_UID` / `VT_GID` | `1000` | user the container runs as, so `./outputs` files belong to you (`id -u` / `id -g`) |
| `VT_THREADS` | `4` | sets `OMP_NUM_THREADS` |
| `VT_CPUS` | `4` | CPU limit |
| `VT_MEMORY` | `3g` | memory limit |

## Sizing

Measured outside Docker on the development laptop (i5-13420H, shared with other work), so treat
these as rough:

| Workload | Observed |
|---|---|
| `run`, YOLO11n @ 960, PyTorch CPU, 1080p, 150 frames, writing video | ~7.9 FPS, end-to-end p50 113 ms/frame, peak RSS ~800 MB (`outputs/readme_demo/summary.json`) |
| same clip, YOLO11n @ 640, ONNX Runtime CPU, no video, 60 frames | detector p50 ~58 ms/frame |

1 to 1.5 GB of memory is enough for YOLO11n/s at 640 to 960 px; 1280 px inputs and `--batch > 1`
need more, and the default 3 GB limit leaves headroom. The detector is about 78% of per-frame time
(88 of 113 ms at p50), and throughput scales with cores up to about `OMP_NUM_THREADS`, so set
`--cpus` and `VT_THREADS` together. To go faster: smaller `--imgsz`, ONNX weights, `--stride 2`,
or `--no-video` (saves about 15 ms/frame of render and encode at 1080p).

## Where it fits

Any CPU Docker host (VM, bare metal, a Kubernetes job) works as is. Treat each `run` as a batch
job. Expect single-digit FPS at 1080p on a laptop-class CPU, so full-HD streams aren't real time
without a GPU.

Serverless platforms (Cloud Run, Lambda, App Runner) are a poor fit: there's no HTTP server, the
image is large, so cold starts are slow, and video jobs outlast typical request timeouts.

Hosts with 512 MB to 1 GB of RAM will likely OOM with PyTorch weights; ONNX at 640 px has the best
chance.

Not tried: ARM64 builds (PyTorch CPU and ONNX Runtime both ship aarch64 wheels, so
`docker buildx build --platform linux/arm64` should work), webcams (`--device /dev/video0`) and
RTSP/HTTP streams inside the container.

MOT17 is CC BY-NC-SA 3.0 (non-commercial), so don't bake it into images you distribute.

## GPU

There is no GPU image and the following hasn't been built. On the development laptop's RTX 4050,
YOLO11s @ 1280 ran at a detector p50 of ~33 ms outside Docker (`outputs/smoke/summary.json`),
which is what real-time 1080p would need. A GPU variant would need:

* a CUDA runtime base (e.g. `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` or a `pytorch/pytorch`
  CUDA image) with torch from the `cu124` index;
* `onnxruntime-gpu` instead of `onnxruntime`, otherwise the ONNX backend falls back to CPU with a
  warning;
* the NVIDIA Container Toolkit on the host and `--gpus all` (compose:
  `deploy.resources.reservations.devices`).
