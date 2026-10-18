"""Image pre/post-processing for YOLO models, independent of torch.

Mirrors ultralytics' LetterBox + scale_boxes + NMS so that the ONNX Runtime
path produces the same boxes as the PyTorch path for the same input tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass
class LetterboxInfo:
    gain: float
    pad_left: int
    pad_top: int
    orig_shape: tuple[int, int]  # (h, w)
    input_shape: tuple[int, int]  # (h, w)


def letterbox_shape(orig_hw: tuple[int, int], imgsz: int, rect: bool, stride: int = 32) -> tuple[int, int]:
    """Target (h, w): square imgsz x imgsz, or the minimal stride-aligned rectangle when ``rect``."""
    if not rect:
        return imgsz, imgsz
    h, w = orig_hw
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    return new_h + (imgsz - new_h) % stride, new_w + (imgsz - new_w) % stride


def letterbox(img: np.ndarray, new_shape: tuple[int, int]) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio, then pad (value 114) to ``new_shape`` = (h, w), centred."""
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    new_unpad_w, new_unpad_h = round(w * r), round(h * r)
    dw, dh = (nw - new_unpad_w) / 2, (nh - new_unpad_h) / 2
    if (w, h) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return img, LetterboxInfo(r, left, top, (h, w), (nh, nw))


def to_tensor(images: list[np.ndarray], dtype=np.float32) -> np.ndarray:
    """BGR HWC uint8 list -> RGB NCHW in [0, 1]."""
    x = np.stack(images)[..., ::-1].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x, dtype=dtype) / dtype(255.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    """Greedy NMS (same rule as torchvision: suppress when IoU > iou_thres). Returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    pred: np.ndarray,
    info: LetterboxInfo,
    conf_thres: float,
    iou_thres: float,
    classes: list[int] | None,
    max_det: int = 300,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw YOLOv8/11 head output (4 + nc, N) for one image -> (xyxy, conf, cls) in original pixels.

    Same steps as ultralytics' non_max_suppression (multi_label=False):
    best class per anchor, conf > thres, class filter, class-offset NMS.
    """
    pred = pred.astype(np.float32, copy=False)
    cls_scores = pred[4:]
    conf = cls_scores.max(0)
    cand = conf > conf_thres
    if not cand.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    cls_id = cls_scores[:, cand].argmax(0)
    conf = conf[cand]
    xywh = pred[:4, cand].T
    if classes is not None:
        m = np.isin(cls_id, classes)
        cls_id, conf, xywh = cls_id[m], conf[m], xywh[m]
    if len(conf) > max_nms:
        top = np.argsort(-conf, kind="stable")[:max_nms]
        cls_id, conf, xywh = cls_id[top], conf[top], xywh[top]
    boxes = np.empty_like(xywh)
    boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
    boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
    keep = nms(boxes + (cls_id[:, None] * max_wh), conf, iou_thres, max_det)
    boxes, conf, cls_id = boxes[keep], conf[keep], cls_id[keep]
    # Undo the letterbox and clip to the frame.
    boxes[:, [0, 2]] -= info.pad_left
    boxes[:, [1, 3]] -= info.pad_top
    boxes /= info.gain
    h, w = info.orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes.astype(np.float32), conf.astype(np.float32), cls_id.astype(np.int64)
