"""The ONNX path re-implements YOLO pre/post-processing in numpy; check it against ultralytics."""

from pathlib import Path

import numpy as np
import pytest

from visiontrack.preprocess import letterbox, letterbox_shape, nms, postprocess, to_tensor

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")


@pytest.mark.parametrize("hw", [(1080, 1920), (480, 640), (720, 1280), (333, 517), (1000, 300)])
@pytest.mark.parametrize("rect", [False, True])
def test_letterbox_matches_ultralytics(hw, rect):
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (*hw, 3), dtype=np.uint8)
    ours, info = letterbox(img, letterbox_shape(hw, 640, rect))
    ref = LetterBox((640, 640), auto=rect, stride=32)(image=img)
    assert ours.shape == ref.shape
    np.testing.assert_array_equal(ours, ref)
    assert info.orig_shape == hw


def test_nms_matches_torchvision():
    import torchvision

    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 500, (400, 2))
    wh = rng.uniform(10, 120, (400, 2))
    boxes = np.concatenate([xy, xy + wh], 1).astype(np.float32)
    scores = rng.uniform(0, 1, 400).astype(np.float32)
    for thr in (0.3, 0.5, 0.7):
        ref = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), thr).numpy()
        np.testing.assert_array_equal(nms(boxes, scores, thr, max_det=10_000), ref)
    assert len(nms(boxes, scores, 0.7, max_det=5)) == 5


def test_postprocess_matches_ultralytics_nms():
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    rng = np.random.default_rng(2)
    n, nc = 8400, 80
    pred = np.zeros((4 + nc, n), np.float32)
    pred[0] = rng.uniform(0, 640, n)
    pred[1] = rng.uniform(0, 640, n)
    pred[2:4] = rng.uniform(5, 200, (2, n))
    pred[4:] = rng.uniform(0, 1, (nc, n)) ** 12  # mostly low scores, a few high ones
    img = np.zeros((1080, 1920, 3), np.uint8)
    _, info = letterbox(img, (640, 640))

    for classes in (None, [0, 2]):
        boxes, conf, cls = postprocess(pred, info, conf_thres=0.25, iou_thres=0.7, classes=classes)
        ref = non_max_suppression(torch.from_numpy(pred[None]), 0.25, 0.7, classes=classes, max_det=300)[0]
        ref[:, :4] = scale_boxes((640, 640), ref[:, :4], img.shape[:2])
        ref = ref.numpy()
        assert len(boxes) == len(ref) > 0
        np.testing.assert_allclose(boxes, ref[:, :4], atol=1e-3)
        np.testing.assert_allclose(conf, ref[:, 4], atol=1e-6)
        np.testing.assert_array_equal(cls, ref[:, 5].astype(int))


def test_to_tensor_layout():
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0] = 255  # blue channel in BGR
    x = to_tensor([img])
    assert x.shape == (1, 3, 4, 6) and x.dtype == np.float32
    assert x[0, 2].max() == 1.0 and x[0, 0].max() == 0.0  # blue ends up last (RGB)


@pytest.mark.slow
def test_onnx_raw_output_parity(tmp_path):
    """Same input tensor through PyTorch and ONNX Runtime -> numerically equivalent head output."""
    weights = Path("models/yolo11n.pt")
    if not weights.exists():
        pytest.skip("models/yolo11n.pt not downloaded")
    from visiontrack.bench.export import export_onnx, raw_outputs_onnx, raw_outputs_torch

    onnx_path = export_onnx("yolo11n.pt", 320, out_dir=tmp_path)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    boxed, _ = letterbox(img, (320, 320))
    x = to_tensor([boxed])
    a = raw_outputs_torch("yolo11n.pt", x)
    b = raw_outputs_onnx(onnx_path, x)
    assert a.shape == b.shape
    assert np.abs(a[:, :4] - b[:, :4]).max() < 0.05  # pixels
    assert np.abs(a[:, 4:] - b[:, 4:]).max() < 1e-3  # class probabilities
