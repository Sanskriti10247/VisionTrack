"""Video ingestion (files, image sequences, webcams/streams) and annotated video output."""

from __future__ import annotations

import configparser
import logging
import math
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
DEFAULT_FPS = 30.0


class VideoOpenError(RuntimeError):
    pass


@dataclass
class Frame:
    index: int  # index in the source (0-based)
    timestamp_s: float
    image: np.ndarray  # BGR uint8


class VideoSource:
    """Iterates frames from a video file, an image directory, or a live camera.

    ``stride`` keeps every Nth frame; skipped frames are grabbed but not
    decoded where the backend allows it. Timestamps are derived from the
    frame index and the container FPS for files (container timestamps are
    unreliable across codecs) and from wall-clock time for live sources.
    """

    def __init__(
        self,
        source: str | int | Path,
        stride: int = 1,
        start_frame: int = 0,
        max_frames: int | None = None,
        fps_override: float | None = None,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.source = source
        self.stride = stride
        self.start_frame = start_frame
        self.max_frames = max_frames
        self._cap: cv2.VideoCapture | None = None
        self._images: list[Path] | None = None
        self.is_live = False
        self.fps_estimated = False

        src = str(source)
        path = Path(src)
        if isinstance(source, int) or src.isdigit():
            self.is_live = True
            self._open_capture(int(src))
        elif path.is_dir():
            self._open_image_dir(path)
        elif src.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            self.is_live = True
            self._open_capture(src)
        else:
            if not path.exists():
                raise VideoOpenError(f"video not found: {src}")
            self._open_capture(src)

        if fps_override:
            self.fps = float(fps_override)
            self.fps_estimated = False

    # ------------------------------------------------------------------ open
    def _open_capture(self, src: str | int) -> None:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise VideoOpenError(f"could not open video source: {src}")
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise VideoOpenError(f"could not decode any frame from: {src}")
        self.height, self.width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0 or fps > 1000:
            fps, self.fps_estimated = DEFAULT_FPS, True
        self.fps = float(fps)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_count: int | None = n if n > 0 else None
        if not self.is_live:
            # Rewind so iteration starts at frame 0; some backends cannot seek, so reopen instead.
            cap.release()
            cap = cv2.VideoCapture(src)
        self._cap = cap

    def _open_image_dir(self, path: Path) -> None:
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            raise VideoOpenError(f"no images found in {path}")
        first = cv2.imread(str(images[0]))
        if first is None:
            raise VideoOpenError(f"could not read image {images[0]}")
        self.height, self.width = first.shape[:2]
        self._images = images
        self.frame_count = len(images)
        self.fps = self._seqinfo_fps(path) or DEFAULT_FPS
        self.fps_estimated = self._seqinfo_fps(path) is None

    @staticmethod
    def _seqinfo_fps(img_dir: Path) -> float | None:
        """MOTChallenge sequences store their frame rate in ../seqinfo.ini."""
        ini = img_dir.parent / "seqinfo.ini"
        if not ini.exists():
            return None
        cp = configparser.ConfigParser()
        cp.read(ini)
        try:
            return float(cp["Sequence"]["frameRate"])
        except (KeyError, ValueError):
            return None

    # ----------------------------------------------------------------- iterate
    @property
    def output_fps(self) -> float:
        return self.fps / self.stride

    @property
    def expected_frames(self) -> int | None:
        if self.frame_count is None:
            return self.max_frames
        n = max(0, math.ceil((self.frame_count - self.start_frame) / self.stride))
        return min(n, self.max_frames) if self.max_frames else n

    def __iter__(self) -> Iterator[Frame]:
        if self._images is not None:
            yield from self._iter_images()
        else:
            yield from self._iter_capture()

    def _iter_images(self) -> Iterator[Frame]:
        emitted = 0
        assert self._images is not None
        for idx in range(self.start_frame, len(self._images), self.stride):
            img = cv2.imread(str(self._images[idx]))
            if img is None:
                log.warning("skipping unreadable image %s", self._images[idx])
                continue
            yield Frame(idx, idx / self.fps, img)
            emitted += 1
            if self.max_frames and emitted >= self.max_frames:
                return

    def _iter_capture(self) -> Iterator[Frame]:
        cap = self._cap
        assert cap is not None
        emitted = 0
        idx = -1
        t0 = time.monotonic()
        while True:
            idx += 1
            keep = idx >= self.start_frame and (idx - self.start_frame) % self.stride == 0
            if not keep:
                if not cap.grab():
                    return
                continue
            ok, img = cap.read()
            if not ok or img is None:
                return
            ts = (time.monotonic() - t0) if self.is_live else idx / self.fps
            yield Frame(idx, ts, img)
            emitted += 1
            if self.max_frames and emitted >= self.max_frames:
                return

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def info(self) -> dict:
        return {
            "source": str(self.source),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "fps_estimated": self.fps_estimated,
            "frame_count": self.frame_count,
            "stride": self.stride,
            "is_live": self.is_live,
        }


class VideoWriter:
    """Writes BGR frames with OpenCV, then (optionally) re-encodes to H.264.

    OpenCV wheels ship without an H.264 encoder, and browsers cannot play
    MPEG-4 Part 2, so when ffmpeg is available the file is transcoded so the
    dashboard can play it inline.
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], browser_compatible: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.browser_compatible = browser_compatible and shutil.which("ffmpeg") is not None
        self._raw_path = self.path.with_suffix(".raw.mp4") if self.browser_compatible else self.path
        self._writer = cv2.VideoWriter(
            str(self._raw_path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), size
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self._raw_path}")
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self.frames += 1

    def close(self) -> Path:
        self._writer.release()
        if self.browser_compatible and self._raw_path != self.path:
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(self._raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "26",
                "-movflags",
                "+faststart",
                str(self.path),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=600)
                self._raw_path.unlink(missing_ok=True)
            except (subprocess.SubprocessError, OSError) as e:
                log.warning("ffmpeg transcode failed (%s); keeping mp4v output", e)
                self._raw_path.replace(self.path)
        return self.path
