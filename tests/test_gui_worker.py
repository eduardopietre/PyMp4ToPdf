import pytest

from mp4_to_pdf_gui import Mp4ToPdfWorker
from tests.helpers import FakeVideoCapture, RecordingQueue, assert_valid_pdf, bgr_frame


def _worker(tmp_path, n_frame=1, diff=0.90, ssim=0.90):
    return Mp4ToPdfWorker(
        RecordingQueue(),
        str(tmp_path / "in.mp4"),
        str(tmp_path / "out.pdf"),
        n_frame,
        diff,
        ssim,
    )


def _patch_frames(monkeypatch, frames):
    capture = FakeVideoCapture(frames)
    monkeypatch.setattr("cv2.VideoCapture", lambda _path: capture)
    return capture


def test_worker_constants():
    assert Mp4ToPdfWorker.UPDATE_READING == 1
    assert Mp4ToPdfWorker.UPDATE_DIFF == 2
    assert Mp4ToPdfWorker.UPDATE_SMI == 3
    assert Mp4ToPdfWorker.DONE == 4


def test_run_calls_convert(tmp_path, monkeypatch):
    worker = _worker(tmp_path)
    called = {"n": 0}
    monkeypatch.setattr(worker, "convert", lambda: called.__setitem__("n", called["n"] + 1))
    worker.run()
    assert called["n"] == 1


def test_convert_emits_done(tmp_path, monkeypatch, red, blue):
    _patch_frames(monkeypatch, [red, blue])
    worker = _worker(tmp_path)
    worker.convert()
    assert worker.queue.items[-1] == (Mp4ToPdfWorker.DONE, 0)


def test_convert_writes_pdf_for_scene_change(tmp_path, monkeypatch, red, blue):
    _patch_frames(monkeypatch, [red, red.copy(), blue])
    worker = _worker(tmp_path)
    worker.convert()
    assert_valid_pdf(worker.out, page_count=1)


def test_convert_identical_frames_raise_on_empty_pdf(tmp_path, monkeypatch, red):
    _patch_frames(monkeypatch, [red, red.copy()])
    worker = _worker(tmp_path)
    with pytest.raises(IndexError):
        worker.convert()
    assert (Mp4ToPdfWorker.DONE, 0) not in worker.queue.items


def test_convert_progress_includes_all_stages(tmp_path, monkeypatch, red, blue, green):
    _patch_frames(monkeypatch, [red, blue, green])
    worker = _worker(tmp_path)
    worker.convert()
    codes = [code for code, _ in worker.queue.items]
    assert Mp4ToPdfWorker.UPDATE_READING in codes
    assert Mp4ToPdfWorker.UPDATE_DIFF in codes
    assert Mp4ToPdfWorker.UPDATE_SMI in codes
    assert codes[-1] == Mp4ToPdfWorker.DONE


def test_convert_with_fake_capture(tmp_path, monkeypatch):
    red_bgr = bgr_frame(32, 48, (0, 0, 255))
    blue_bgr = bgr_frame(32, 48, (255, 0, 0))
    _patch_frames(monkeypatch, [red_bgr, blue_bgr])
    worker = _worker(tmp_path)
    worker.convert()
    assert worker.queue.items[-1] == (Mp4ToPdfWorker.DONE, 0)
    assert_valid_pdf(worker.out, page_count=1)


def test_thread_is_not_daemon_by_default(tmp_path):
    worker = _worker(tmp_path)
    assert worker.daemon is False


@pytest.mark.parametrize("diff,ssim", [(0.0, 0.0), (0.5, 0.5), (0.90, 0.90), (1.0, 1.0)])
def test_worker_keeps_thresholds(tmp_path, diff, ssim):
    worker = _worker(tmp_path, diff=diff, ssim=ssim)
    assert worker.diff_threshold == diff
    assert worker.ssim_threshold == ssim
