# CPU image for the visiontrack CLI. Weights are mounted at /app/models, see docs/deployment.md.
#   docker build -f docker/Dockerfile -t visiontrack:cpu .

ARG PYTHON_IMAGE=python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

FROM ${PYTHON_IMAGE} AS builder

ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# cpu torch first, otherwise ultralytics pulls the multi-GB CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

WORKDIR /src
COPY pyproject.toml README.md ./
COPY visiontrack ./visiontrack
RUN pip install ".[onnx]"

FROM ${PYTHON_IMAGE} AS runtime

# opencv-python needs libGL and glib at import time
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 visiontrack \
 && useradd --system --uid 10001 --gid visiontrack --home-dir /app --shell /usr/sbin/nologin visiontrack

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=visiontrack:visiontrack configs ./configs
# ultralytics' predictor creates runs/ under the cwd even when it saves nothing.
# 1777 so it still works when run with --user $(id -u)
RUN mkdir -p /app/models /app/data /app/outputs /app/runs \
 && chown -R visiontrack:visiontrack /app \
 && chmod 1777 /app/runs /app/outputs

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp \
    YOLO_OFFLINE=true \
    OMP_NUM_THREADS=4

USER visiontrack

# no server to probe, so just check the package and backends import
HEALTHCHECK --interval=30s --timeout=20s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import visiontrack.pipeline, onnxruntime, cv2"]

ENTRYPOINT ["visiontrack"]
CMD ["--help"]
