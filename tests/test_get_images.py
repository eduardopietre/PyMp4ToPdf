import numpy as np
import pytest

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker
from tests.helpers import FakeVideoCapture, RecordingQueue, bgr_frame


def _cli(n_frame=1, lim=None, verbose=False, infile="in.mp4"):
    return Mp4ToPdf(infile, "out.pdf", n_frame, lim, 0.90, 0.90, verbose=verbose)


def _gui(n_frame=1, infile="in.mp4"):
    return Mp4ToPdfWorker(RecordingQueue(), infile, "out.pdf", n_frame, 0.90, 0.90)


def _patch_capture(monkeypatch, frames, fps=30.0):
    capture = FakeVideoCapture(frames, fps=fps)

    def factory(_path):
        return capture

    monkeypatch.setattr("cv2.VideoCapture", factory)
    return capture


def test_cli_converts_bgr_to_rgb(monkeypatch):
    bgr = bgr_frame(16, 16, (0, 0, 255))
    _patch_capture(monkeypatch, [bgr])
    images = _cli().get_images()
    assert len(images) == 1
    np.testing.assert_array_equal(images[0][0, 0], (255, 0, 0))


def test_gui_converts_bgr_to_rgb(monkeypatch):
    bgr = bgr_frame(16, 16, (255, 0, 0))
    _patch_capture(monkeypatch, [bgr])
    images = _gui().get_images()
    np.testing.assert_array_equal(images[0][0, 0], (0, 0, 255))


@pytest.mark.parametrize("n_frame, expected_indexes", [
    (1, [0, 1, 2, 3, 4]),
    (2, [0, 2, 4]),
    (3, [0, 3]),
    (4, [0, 4]),
    (5, [0]),
    (10, [0]),
])
def test_cli_frame_skip(monkeypatch, n_frame, expected_indexes):
    frames = [bgr_frame(8, 8, (i, i, i)) for i in range(5)]
    _patch_capture(monkeypatch, frames)
    images = _cli(n_frame=n_frame).get_images()
    got = [int(img[0, 0, 0]) for img in images]
    assert got == expected_indexes


@pytest.mark.parametrize("n_frame, expected_indexes", [
    (1, [0, 1, 2, 3]),
    (2, [0, 2]),
    (3, [0, 3]),
])
def test_gui_frame_skip(monkeypatch, n_frame, expected_indexes):
    frames = [bgr_frame(8, 8, (i, 0, 0)) for i in range(4)]
    _patch_capture(monkeypatch, frames)
    images = _gui(n_frame=n_frame).get_images()
    got = [int(img[0, 0, 2]) for img in images]
    assert got == expected_indexes


def test_empty_capture_returns_no_images(monkeypatch):
    _patch_capture(monkeypatch, [])
    assert _cli().get_images() == []
    assert _gui().get_images() == []


def test_cli_lim_stops_after_count_exceeds_limit(monkeypatch):
    frames = [bgr_frame(8, 8, (i, 0, 0)) for i in range(20)]
    _patch_capture(monkeypatch, frames)
    images = _cli(n_frame=1, lim=3).get_images()
    assert [int(img[0, 0, 2]) for img in images] == [0, 1, 2, 3]


def test_cli_lim_none_reads_all(monkeypatch):
    frames = [bgr_frame(8, 8, (i, 0, 0)) for i in range(6)]
    _patch_capture(monkeypatch, frames)
    images = _cli(n_frame=1, lim=None).get_images()
    assert len(images) == 6


def test_cli_lim_zero_is_falsy_and_does_not_stop(monkeypatch):
    frames = [bgr_frame(8, 8, (i, 0, 0)) for i in range(4)]
    _patch_capture(monkeypatch, frames)
    images = _cli(n_frame=1, lim=0).get_images()
    assert len(images) == 4


def test_gui_emits_reading_progress_and_completion(monkeypatch):
    frames = [bgr_frame(8, 8, (i, 0, 0)) for i in range(3)]
    _patch_capture(monkeypatch, frames)
    worker = _gui(n_frame=1)
    worker.get_images()
    assert worker.queue.items[-1] == (Mp4ToPdfWorker.UPDATE_READING, 1000)
    assert all(code == Mp4ToPdfWorker.UPDATE_READING for code, _ in worker.queue.items)
    assert len(worker.queue.items) == 4


def test_gui_progress_uses_video_length(monkeypatch):
    frames = [bgr_frame(8, 8, (0, 0, 0)) for _ in range(10)]
    _patch_capture(monkeypatch, frames)
    worker = _gui(n_frame=5)
    worker.get_images()
    first = worker.queue.items[0]
    assert first[0] == Mp4ToPdfWorker.UPDATE_READING
    assert first[1] == 600


def test_released_capture_is_closed(monkeypatch):
    frames = [bgr_frame(8, 8, (1, 2, 3))]
    capture = _patch_capture(monkeypatch, frames)
    _cli().get_images()
    assert capture.opened is False


def test_cli_logs_video_info_when_verbose(monkeypatch, capsys):
    frames = [bgr_frame(8, 8, (0, 0, 0)) for _ in range(2)]
    _patch_capture(monkeypatch, frames, fps=25.0)
    _cli(verbose=True).get_images()
    out = capsys.readouterr().out
    assert "FPS: 25.0" in out
    assert "Lenght: 2 frames" in out
    assert "Duration:" in out
