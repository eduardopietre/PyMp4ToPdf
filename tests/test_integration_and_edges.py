import pytest

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker, to_per_mile
from tests.helpers import FakeVideoCapture, RecordingQueue, assert_valid_pdf, bgr_frame, rgb_frame, write_video


@pytest.mark.integration
def test_cli_reads_real_mp4_and_converts_color(tmp_path, monkeypatch):
    red_bgr = bgr_frame(32, 48, (0, 0, 255))
    blue_bgr = bgr_frame(32, 48, (255, 0, 0))
    path = tmp_path / "clip.mp4"
    write_video(path, [red_bgr, red_bgr, blue_bgr, blue_bgr], fps=10)
    converter = Mp4ToPdf(str(path), str(tmp_path / "out.pdf"), 1, None, 0.90, 0.90, verbose=False)
    images = converter.get_images()
    assert len(images) >= 1
    first = images[0][16, 24]
    assert first[2] > first[0]


@pytest.mark.integration
def test_gui_full_convert_from_real_video(tmp_path):
    red_bgr = bgr_frame(32, 48, (0, 0, 255))
    blue_bgr = bgr_frame(32, 48, (255, 0, 0))
    path = tmp_path / "clip.mp4"
    write_video(path, [red_bgr, blue_bgr], fps=10)
    worker = Mp4ToPdfWorker(RecordingQueue(), str(path), str(tmp_path / "out.pdf"), 1, 0.90, 0.90)
    worker.convert()
    assert_valid_pdf(tmp_path / "out.pdf", page_count=1)


@pytest.mark.parametrize("h,w", [(7, 7), (7, 64), (64, 7), (9, 11)])
def test_minimum_ssim_spatial_size(h, w):
    a = rgb_frame(h, w, (255, 128, 0))
    b = rgb_frame(h, w, (0, 128, 255))
    worker = Mp4ToPdfWorker(RecordingQueue(), "in.mp4", "out.pdf", 1, 0.90, 0.90)
    fails = worker.structural_similarity_filter([[a, b]])
    assert len(fails) == 1


@pytest.mark.parametrize("color_a,color_b,expect_diff", [
    ((0, 0, 0), (0, 0, 0), False),
    ((0, 0, 0), (1, 0, 0), True),
    ((255, 255, 255), (255, 255, 255), False),
    ((10, 20, 30), (10, 20, 31), True),
])
def test_diff_filter_pixel_sensitivity(color_a, color_b, expect_diff):
    a = rgb_frame(16, 16, color_a)
    b = rgb_frame(16, 16, color_b)
    converter = Mp4ToPdf("in.mp4", "out.pdf", 1, None, 0.999, 0.90, verbose=False)
    pairs = converter.diff_filter([a, b])
    assert bool(pairs) is expect_diff


def test_to_per_mile_used_by_worker_progress():
    length = 8
    values = [to_per_mile(i + 1, length) for i in range(length)]
    assert values[0] < values[-1]
    assert values[-1] == 1000


def test_worker_does_not_put_done_if_capture_raises(tmp_path, monkeypatch):
    worker = Mp4ToPdfWorker(RecordingQueue(), str(tmp_path / "in.mp4"), str(tmp_path / "out.pdf"), 1, 0.90, 0.90)

    def boom(*_args, **_kwargs):
        raise RuntimeError("cannot read")

    monkeypatch.setattr("cv2.VideoCapture", boom)
    with pytest.raises(RuntimeError):
        worker.convert()
    assert worker.queue.items == []


def test_cli_get_images_uses_infile_path(tmp_path, monkeypatch):
    seen = {}

    def factory(path):
        seen["path"] = path
        return FakeVideoCapture([])

    monkeypatch.setattr("cv2.VideoCapture", factory)
    converter = Mp4ToPdf(str(tmp_path / "custom.mp4"), str(tmp_path / "out.pdf"), 1, None, 0.90, 0.90)
    converter.get_images()
    assert seen["path"] == str(tmp_path / "custom.mp4")
