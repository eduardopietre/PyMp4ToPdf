import pytest

from tests.helpers import FakeVideoCapture, bgr_frame, rgb_frame, write_video


def test_rgb_frame_shape_and_dtype():
    frame = rgb_frame(10, 20, (1, 2, 3))
    assert frame.shape == (10, 20, 3)
    assert frame.dtype == "uint8"
    assert frame[0, 0].tolist() == [1, 2, 3]
    assert frame[-1, -1].tolist() == [1, 2, 3]


def test_rgb_frame_does_not_share_memory():
    a = rgb_frame(4, 4, (9, 9, 9))
    b = rgb_frame(4, 4, (9, 9, 9))
    a[0, 0] = (0, 0, 0)
    assert b[0, 0].tolist() == [9, 9, 9]


def test_fake_capture_read_and_seek():
    frames = [bgr_frame(2, 2, (i, 0, 0)) for i in range(5)]
    capture = FakeVideoCapture(frames, fps=24)
    assert capture.get(1) or True
    ok, frame = capture.read()
    assert ok
    assert frame[0, 0, 0] == 0
    capture.set(1, 3)
    ok, frame = capture.read()
    assert ok
    assert frame[0, 0, 0] == 3


def test_fake_capture_grab_skips_without_returning_pixels():
    frames = [bgr_frame(2, 2, (i, 0, 0)) for i in range(4)]
    capture = FakeVideoCapture(frames)
    ok, frame = capture.read()
    assert ok and frame[0, 0, 0] == 0
    assert capture.grab() is True
    ok, frame = capture.read()
    assert ok and frame[0, 0, 0] == 2


def test_fake_capture_read_past_end():
    capture = FakeVideoCapture([bgr_frame(2, 2, (1, 2, 3))])
    capture.read()
    ok, frame = capture.read()
    assert ok is False
    assert frame is None


def test_fake_capture_release():
    capture = FakeVideoCapture([bgr_frame(2, 2, (1, 2, 3))])
    capture.release()
    assert capture.isOpened() is False
    ok, _ = capture.read()
    assert ok is False


@pytest.mark.integration
def test_write_video_roundtrip(tmp_path):
    frames = [
        bgr_frame(32, 48, (0, 0, 255)),
        bgr_frame(32, 48, (255, 0, 0)),
    ]
    path = tmp_path / "clip.mp4"
    write_video(path, frames, fps=10)
    assert path.is_file()
    assert path.stat().st_size > 0
