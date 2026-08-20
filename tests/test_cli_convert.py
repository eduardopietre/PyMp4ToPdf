from types import SimpleNamespace

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


def test_stores_constructor_arguments(tmp_path):
    converter = _make(tmp_path, n_frame=24, lim=10, diff_threshold=0.5, ssim_threshold=0.7, verbose=True)
    assert converter.n_frame == 24
    assert converter.lim == 10
    assert converter.diff_threshold == 0.5
    assert converter.ssim_threshold == 0.7
    assert converter.verbose is True


def test_convert_requires_module_level_args(tmp_path, monkeypatch, red, blue):
    converter = _make(tmp_path)
    monkeypatch.setattr(converter, "get_images", lambda: [red, blue])
    with pytest.raises(NameError):
        converter.convert()


def test_convert_uses_args_infile_only_for_logging(tmp_path, monkeypatch, red, blue):
    converter = _make(tmp_path, verbose=True)
    monkeypatch.setattr("mp4_to_pdf.args", SimpleNamespace(infile="logged.mp4"), raising=False)
    monkeypatch.setattr(converter, "get_images", lambda: [red, blue])
    converter.convert()
    assert converter.infile != "logged.mp4"


def test_convert_end_to_end_with_scene_change(tmp_path, monkeypatch, red, blue):
    monkeypatch.setattr("mp4_to_pdf.args", SimpleNamespace(infile="in.mp4"), raising=False)
    converter = _make(tmp_path)
    monkeypatch.setattr(converter, "get_images", lambda: [red, red.copy(), blue, blue.copy()])
    converter.convert()
    assert_valid_pdf(converter.out, page_count=1)


def test_convert_identical_frames_cannot_save_empty_pdf(tmp_path, monkeypatch, red):
    monkeypatch.setattr("mp4_to_pdf.args", SimpleNamespace(infile="in.mp4"), raising=False)
    converter = _make(tmp_path)
    monkeypatch.setattr(converter, "get_images", lambda: [red, red.copy()])
    with pytest.raises(IndexError):
        converter.convert()


def test_convert_logs_pipeline_steps(tmp_path, monkeypatch, capsys, red, blue):
    monkeypatch.setattr("mp4_to_pdf.args", SimpleNamespace(infile="demo.mp4"), raising=False)
    converter = _make(tmp_path, verbose=True)
    monkeypatch.setattr(converter, "get_images", lambda: [red, blue])
    converter.convert()
    out = capsys.readouterr().out
    assert "Reading file demo.mp4" in out
    assert "Read 2 images" in out
    assert "Calculating differences" in out
    assert "Applying structural similarity" in out
    assert "Exporting" in out
    assert "Done." in out


def test_convert_uniques_are_the_new_frames_not_the_first_slide(tmp_path, monkeypatch, red, blue, green):
    monkeypatch.setattr("mp4_to_pdf.args", SimpleNamespace(infile="in.mp4"), raising=False)
    converter = _make(tmp_path)
    monkeypatch.setattr(converter, "get_images", lambda: [red, blue, green])
    captured = {}

    def fake_save(images):
        captured["images"] = images

    monkeypatch.setattr(converter, "save_as_pdf", fake_save)
    converter.convert()
    assert len(captured["images"]) == 2
    np.testing.assert_array_equal(captured["images"][0], blue)
    np.testing.assert_array_equal(captured["images"][1], green)


def test_convert_with_fake_capture(tmp_path, monkeypatch):
    monkeypatch.setattr("mp4_to_pdf.args", SimpleNamespace(infile="in.mp4"), raising=False)
    red_bgr = bgr_frame(32, 48, (0, 0, 255))
    blue_bgr = bgr_frame(32, 48, (255, 0, 0))
    frames = [red_bgr, red_bgr.copy(), blue_bgr, blue_bgr.copy()]
    capture = FakeVideoCapture(frames)

    monkeypatch.setattr("cv2.VideoCapture", lambda _path: capture)
    converter = _make(tmp_path)
    converter.convert()
    assert_valid_pdf(converter.out, page_count=1)


@pytest.mark.parametrize("verbose", [False, True])
def test_convert_respects_verbose_flag(tmp_path, monkeypatch, capsys, red, blue, verbose):
    monkeypatch.setattr("mp4_to_pdf.args", SimpleNamespace(infile="in.mp4"), raising=False)
    converter = _make(tmp_path, verbose=verbose)
    monkeypatch.setattr(converter, "get_images", lambda: [red, blue])
    converter.convert()
    out = capsys.readouterr().out
    if verbose:
        assert "Done." in out
    else:
        assert out == ""
