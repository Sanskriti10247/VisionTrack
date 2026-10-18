"""ONNX export (FP32/FP16 via ultralytics) and INT8 static quantization (ONNX Runtime)."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from visiontrack.preprocess import letterbox, to_tensor

log = logging.getLogger(__name__)

ONNX_DIR = Path("models/onnx")


def onnx_name(weights: str, imgsz: int, precision: str, dynamic: bool) -> str:
    return f"{Path(weights).stem}_{imgsz}_{precision}{'_dynamic' if dynamic else ''}.onnx"


def export_onnx(
    weights: str, imgsz: int, half: bool = False, dynamic: bool = False, out_dir: Path = ONNX_DIR
) -> Path:
    """Exports raw head output (1, 4+nc, anchors); NMS stays outside the graph."""
    from ultralytics import YOLO

    from visiontrack.detection.ultralytics_detector import resolve_weights

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / onnx_name(weights, imgsz, "fp16" if half else "fp32", dynamic)
    if dst.exists():
        return dst
    pt = resolve_weights(weights)
    model = YOLO(pt)
    # FP16 export needs a CUDA device in ultralytics; FP32 export is done on CPU.
    device = 0 if half else "cpu"
    path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=True,
        device=device,
        batch=1,
        nms=False,
    )
    shutil.move(path, dst)
    log.info("exported %s", dst)
    return dst


class _CalibReader:
    """Feeds letterboxed calibration frames to the ORT calibrator."""

    def __init__(self, files: list[Path], input_name: str, imgsz: int) -> None:
        self.files = files
        self.input_name = input_name
        self.imgsz = imgsz
        self._it = iter(files)

    def get_next(self):
        f = next(self._it, None)
        if f is None:
            return None
        img = cv2.imread(str(f))
        boxed, _ = letterbox(img, (self.imgsz, self.imgsz))
        return {self.input_name: to_tensor([boxed])}

    def rewind(self) -> None:
        self._it = iter(self.files)


def calibration_files(roots: list[Path], n: int, seed: int = 0) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        files += sorted(r.rglob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no calibration images under {roots}")
    random.Random(seed).shuffle(files)
    return files[:n]


def head_node_names(model_path: Path, head_prefix: str) -> list[str]:
    import onnx

    m = onnx.load(str(model_path))
    return [n.name for n in m.graph.node if n.name.startswith(head_prefix)]


def quantize_int8(
    fp32_path: Path,
    calib: list[Path],
    imgsz: int,
    exclude_head: bool = True,
    method: str = "minmax",
    out_path: Path | None = None,
) -> Path:
    """Static (calibrated) INT8 quantization in QDQ format: U8 activations, S8 per-channel weights.

    The detection head mixes box regression (pixel-scale values after DFL) and
    class probabilities in one concatenated tensor; a single INT8 scale for
    that tensor destroys box precision, so by default the head (``/model.23/``
    in YOLO11) stays in float. See reports/benchmarks for the measured effect.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    tag = f"int8{'' if exclude_head else '_fullgraph'}{'' if method == 'minmax' else '_' + method}"
    out_path = out_path or fp32_path.with_name(fp32_path.name.replace("fp32", tag))
    if out_path.exists():
        return out_path
    prep = fp32_path.with_name(fp32_path.stem + "_prep.onnx")
    quant_pre_process(str(fp32_path), str(prep), skip_symbolic_shape=False)
    input_name = ort.InferenceSession(str(prep), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    exclude = head_node_names(prep, "/model.23/") if exclude_head else []
    quantize_static(
        str(prep),
        str(out_path),
        _CalibReader(calib, input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method={
            "minmax": CalibrationMethod.MinMax,
            "percentile": CalibrationMethod.Percentile,
            "entropy": CalibrationMethod.Entropy,
        }[method],
        nodes_to_exclude=exclude,
    )
    prep.unlink(missing_ok=True)
    _copy_metadata(fp32_path, out_path)
    log.info("quantized %s (%d head nodes kept in float)", out_path, len(exclude))
    return out_path


def _copy_metadata(src: Path, dst: Path) -> None:
    import onnx

    s = onnx.load(str(src))
    d = onnx.load(str(dst))
    existing = {p.key for p in d.metadata_props}
    for p in s.metadata_props:
        if p.key not in existing:
            d.metadata_props.append(p)
    onnx.save(d, str(dst))


def raw_outputs_torch(weights: str, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Raw head output of the PyTorch model for a preprocessed NCHW float32 batch."""
    import torch
    from ultralytics import YOLO

    from visiontrack.detection.ultralytics_detector import resolve_weights

    model = YOLO(resolve_weights(weights)).model.to(device).eval()
    model.fuse() if hasattr(model, "fuse") else None
    with torch.no_grad():
        y = model(torch.from_numpy(x).to(device))
    y = y[0] if isinstance(y, (list, tuple)) else y
    return y.float().cpu().numpy()


def raw_outputs_onnx(path: Path, x: np.ndarray, providers: list[str] | None = None) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    return sess.run(None, {inp.name: x.astype(dtype)})[0].astype(np.float32)
