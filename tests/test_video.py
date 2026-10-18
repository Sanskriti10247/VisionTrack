from pathlib import Path

import cv2
import numpy as np
import pytest

from visiontrack.io.video import VideoOpenError, VideoSource, VideoWriter


def test_reads_all_frames_with_timestamps(tiny_video):
    src = VideoSource(tiny_video)
    frames = list(src)
    assert [f.index for f in frames] == list(range(30))
    assert src.fps == pytest.approx(10.0)
    assert frames[5].timestamp_s == pytest.approx(0.5)
    assert frames[0].image.shape == (120, 160, 3)


def test_stride_start_and_max_frames(tiny_video):
    src = VideoSource(tiny_video, stride=3, start_frame=2, max_frames=4)
    assert [f.index for f in src] == [2, 5, 8, 11]
    assert src.output_fps == pytest.approx(10.0 / 3)
    assert src.expected_frames == 4


def test_strided_frames_are_the_right_frames(tiny_video):
    """Skipped frames are grabbed, not decoded: make sure we still land on the right content."""
    full = {f.index: f.image for f in VideoSource(tiny_video)}
    for f in VideoSource(tiny_video, stride=4):
        assert np.abs(f.image.astype(int) - full[f.index].astype(int)).mean() < 1.0


def test_image_directory_uses_seqinfo_fps(tmp_path: Path):
    img_dir = tmp_path / "seq" / "img1"
    img_dir.mkdir(parents=True)
    for i in range(1, 6):
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), np.full((20, 30, 3), i * 10, np.uint8))
    (tmp_path / "seq" / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\n")
    src = VideoSource(img_dir, stride=2)
    frames = list(src)
    assert [f.index for f in frames] == [0, 2, 4]
    assert src.fps == 25 and not src.fps_estimated
    assert frames[1].timestamp_s == pytest.approx(2 / 25)


def test_missing_and_invalid_files(tmp_path: Path):
    with pytest.raises(VideoOpenError):
        VideoSource(tmp_path / "nope.mp4")
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"definitely not a video")
    with pytest.raises(VideoOpenError):
        VideoSource(bogus)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(VideoOpenError):
        VideoSource(empty_dir)


def test_writer_roundtrip(tmp_path: Path):
    out = tmp_path / "out.mp4"
    w = VideoWriter(out, 10, (64, 48), browser_compatible=False)
    for _ in range(7):
        w.write(np.zeros((48, 64, 3), np.uint8))
    path = w.close()
    assert path.exists()
    assert len(list(VideoSource(path))) == 7
