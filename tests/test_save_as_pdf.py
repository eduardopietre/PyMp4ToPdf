import numpy as np
import pytest

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker
from tests.helpers import RecordingQueue, assert_valid_pdf, rgb_frame


def _cli(out):
    return Mp4ToPdf("in.mp4", out, 1, None, 0.90, 0.90, verbose=False)


def _gui(out):
    return Mp4ToPdfWorker(RecordingQueue(), "in.mp4", out, 1, 0.90, 0.90)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_save_single_image_creates_one_page_pdf(factory, tmp_path, red):
    out = str(tmp_path / "one.pdf")
    factory(out).save_as_pdf([red])
    assert_valid_pdf(out, page_count=1)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_save_multiple_images_appends_pages(factory, tmp_path, red, blue, green):
    out = str(tmp_path / "multi.pdf")
    factory(out).save_as_pdf([red, blue, green])
    assert_valid_pdf(out, page_count=3)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_save_empty_list_raises(factory, tmp_path):
    with pytest.raises(IndexError):
        factory(str(tmp_path / "empty.pdf")).save_as_pdf([])


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_page_count_matches_input(n, tmp_path):
    frames = [rgb_frame(16, 24, (i * 30, 10, 200 - i * 20)) for i in range(n)]
    out = str(tmp_path / f"{n}.pdf")
    _cli(out).save_as_pdf(frames)
    assert_valid_pdf(out, page_count=n)


def test_saved_pdf_dimensions_match_frame(tmp_path, red):
    out = str(tmp_path / "size.pdf")
    _gui(out).save_as_pdf([red])
    reader = assert_valid_pdf(out, page_count=1)
    box = reader.pages[0].mediabox
    expected_width = red.shape[1] * 72 / 100.0
    expected_height = red.shape[0] * 72 / 100.0
    assert float(box.width) == pytest.approx(expected_width, rel=0.05)
    assert float(box.height) == pytest.approx(expected_height, rel=0.05)


def test_save_does_not_mutate_source_array(tmp_path, red):
    original = red.copy()
    _cli(str(tmp_path / "copy.pdf")).save_as_pdf([red])
    np.testing.assert_array_equal(red, original)
    assert_valid_pdf(tmp_path / "copy.pdf")
