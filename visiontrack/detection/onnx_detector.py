"""YOLO detector on ONNX Runtime with numpy pre/post-processing (no torch dependency)."""

from __future__ import annotations

import ast
import logging
import os
import time

import numpy as np
import psutil

from visiontrack.config import DetectorConfig
from visiontrack.detection.base import Detector
from visiontrack.preprocess import letterbox, letterbox_shape, postprocess, to_tensor
from visiontrack.types import COCO_NAMES, Detections

log = logging.getLogger(__name__)


def _providers(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device.startswith("cuda") or device == "auto":
        if "CUDAExecutionProvider" in available:
            dev_id = int(device.split(":")[1]) if ":" in device else 0
            return [("CUDAExecutionProvider", {"device_id": dev_id}), "CPUExecutionProvider"]  # type: ignore[list-item]
        if device != "auto":
            log.warning("CUDAExecutionProvider not available (install onnxruntime-gpu); using CPU")
    return ["CPUExecutionProvider"]


class OnnxDetector(Detector):
    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            # Lets the CUDA provider find the cuDNN/cuBLAS wheels that ship with torch, if present.
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001 - best effort, CPU still works
                pass
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = int(os.environ.get("VT_ORT_THREADS", "0")) or psutil.cpu_count(logical=False) or 1
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(cfg.weights, so, providers=_providers(cfg.device))
        self.providers = self.session.get_providers()
        self.threads = threads
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32
        b, _, h, w = inp.shape
        self.static_batch = b if isinstance(b, int) else None
        self.static_hw = (h, w) if isinstance(h, int) and isinstance(w, int) else None
        if self.static_hw and self.static_hw != (cfg.imgsz, cfg.imgsz):
            log.info("model input is fixed at %s; ignoring imgsz=%d", self.static_hw, cfg.imgsz)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = COCO_NAMES
        if "names" in meta:
            names = ast.literal_eval(meta["names"])
            self.names = tuple(names[i] for i in range(len(names)))
        self.imgsz = self.static_hw[0] if self.static_hw else cfg.imgsz
        # Dynamic-shape exports can use ultralytics-style minimal rectangles (less padding = less compute).
        self.rect = self.static_hw is None

    def _target_shape(self, img: np.ndarray) -> tuple[int, int]:
        if self.static_hw:
            return self.static_hw
        return letterbox_shape(img.shape[:2], self.imgsz, rect=self.rect)

    def _run(self, images: list[np.ndarray]) -> tuple[list[Detections], float, float, float]:
        t0 = time.perf_counter()
        shape = self._target_shape(images[0])
        boxed, infos = zip(*(letterbox(im, shape) for im in images), strict=True)
        x = to_tensor(list(boxed), self.input_dtype)
        t1 = time.perf_counter()
        pred = self.session.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()
        out = []
        for p, info in zip(pred, infos, strict=True):
            b, c, k = postprocess(
                p, info, self.cfg.conf_threshold, self.cfg.iou_threshold, self.cfg.classes, self.cfg.max_det
            )
            out.append(Detections(b, c, k))
        t3 = time.perf_counter()
        return out, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    def detect_batch(self, images: list[np.ndarray]) -> list[Detections]:
        if not images:
            return []
        t_start = time.perf_counter()
        groups: list[list[int]] = []
        if self.static_batch == 1:
            groups = [[i] for i in range(len(images))]
        else:
            # Batch only frames that letterbox to the same input shape.
            by_shape: dict[tuple[int, int], list[int]] = {}
            for i, im in enumerate(images):
                by_shape.setdefault(self._target_shape(im), []).append(i)
            step = self.static_batch or len(images)
            for idxs in by_shape.values():
                groups += [idxs[j : j + step] for j in range(0, len(idxs), step)]
        results: list[Detections | None] = [None] * len(images)
        pre = inf = post = 0.0
        for g in groups:
            dets, a, b, c = self._run([images[i] for i in g])
            pre, inf, post = pre + a, inf + b, post + c
            for i, d in zip(g, dets, strict=True):
                results[i] = d
        n = len(images)
        self.last_timings = {
            "preprocess_ms": pre / n,
            "inference_ms": inf / n,
            "postprocess_ms": post / n,
            "detector_total_ms": (time.perf_counter() - t_start) * 1000 / n,
        }
        return results  # type: ignore[return-value]

    def info(self) -> dict:
        return {
            **super().info(),
            "providers": self.providers,
            "input_dtype": np.dtype(self.input_dtype).name,
            "static_input_hw": self.static_hw,
            "static_batch": self.static_batch,
            "rect": self.rect,
            "ort_intra_op_threads": self.threads,
        }
