import numpy as np
import pytest

from mp4_to_pdf import Mp4ToPdf
from tests.helpers import FakeVideoCapture, assert_valid_pdf, bgr_frame


def _make(tmp_path, **kwargs):
    params = dict(
        infile=str(tmp_path / "in.mp4"),
        out=str(tmp_path / "out.pdf"),
        n_frame=1,
        lim=None,
        diff_threshold=0.90,
        ssim_threshold=0.90,
        verbose=False,
    )
    params.update(kwargs)
    return Mp4ToPdf(**params)


def _patch_frames(monkeypatch, frames):
    capture = FakeVideoCapture(frames)
    monkeypatch.setattr("cv2.VideoCapture", lambda _path: capture)
    return capture


def test_stores_constructor_arguments(tmp_path):
    converter = _make(tmp_path, n_frame=24, lim=10, diff_threshold=0.5, ssim_threshold=0.7, verbose=True)
    assert converter.n_frame == 24
    assert converter.lim == 10
    assert converter.diff_threshold == 0.5
    assert converter.ssim_threshold == 0.7
    assert converter.verbose is True


def test_convert_does_not_need_module_level_args(tmp_path, monkeypatch, red, blue):
    _patch_frames(monkeypatch, [red, blue])
    converter = _make(tmp_path)
    converter.convert()
    assert_valid_pdf(converter.out, page_count=1)


def test_convert_logs_self_infile(tmp_path, monkeypatch, capsys, red, blue):
    _patch_frames(monkeypatch, [red, blue])
    converter = _make(tmp_path, verbose=True)
    converter.convert()
    out = capsys.readouterr().out
    assert converter.infile in out
    assert "logged.mp4" not in out


def test_convert_end_to_end_with_scene_change(tmp_path, monkeypatch, red, blue):
    _patch_frames(monkeypatch, [red, red.copy(), blue, blue.copy()])
    converter = _make(tmp_path)
    converter.convert()
    assert_valid_pdf(converter.out, page_count=1)


def test_convert_identical_frames_cannot_save_empty_pdf(tmp_path, monkeypatch, red):
    _patch_frames(monkeypatch, [red, red.copy()])
    converter = _make(tmp_path)
    with pytest.raises(IndexError):
        converter.convert()


def test_convert_logs_pipeline_steps(tmp_path, monkeypatch, capsys, red, blue):
    _patch_frames(monkeypatch, [red, blue])
    converter = _make(tmp_path, verbose=True)
    converter.convert()
    out = capsys.readouterr().out
    assert f"Reading file {converter.infile}" in out
    assert "Read 2 images" in out
    assert "Calculating differences" in out
    assert "Applying structural similarity" in out
    assert "Exporting" in out
    assert "Done." in out


def test_convert_uniques_are_the_new_frames_not_the_first_slide(tmp_path, monkeypatch, red, blue, green):
    _patch_frames(monkeypatch, [red, blue, green])
    converter = _make(tmp_path)
    captured = {}

    def fake_save(images):
        captured["images"] = images

    monkeypatch.setattr(converter, "save_as_pdf", fake_save)
    converter.convert()
    assert len(captured["images"]) == 2
    np.testing.assert_array_equal(captured["images"][0], blue)
    np.testing.assert_array_equal(captured["images"][1], green)


def test_convert_with_fake_capture(tmp_path, monkeypatch):
    red_bgr = bgr_frame(32, 48, (0, 0, 255))
    blue_bgr = bgr_frame(32, 48, (255, 0, 0))
    frames = [red_bgr, red_bgr.copy(), blue_bgr, blue_bgr.copy()]
    _patch_frames(monkeypatch, frames)
    converter = _make(tmp_path)
    converter.convert()
    assert_valid_pdf(converter.out, page_count=1)


@pytest.mark.parametrize("verbose", [False, True])
def test_convert_respects_verbose_flag(tmp_path, monkeypatch, capsys, red, blue, verbose):
    _patch_frames(monkeypatch, [red, blue])
    converter = _make(tmp_path, verbose=verbose)
    converter.convert()
    out = capsys.readouterr().out
    if verbose:
        assert "Done." in out
    else:
        assert out == ""
