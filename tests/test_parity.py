import numpy as np
import pytest

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker
from tests.helpers import RecordingQueue, rgb_frame


def _cli():
    return Mp4ToPdf("in.mp4", "out.pdf", 1, None, 0.90, 0.90, verbose=False)


def _gui():
    return Mp4ToPdfWorker(RecordingQueue(), "in.mp4", "out.pdf", 1, 0.90, 0.90)


def _scene_frames():
    red = rgb_frame(32, 48, (200, 20, 20))
    blue = rgb_frame(32, 48, (20, 20, 200))
    green = rgb_frame(32, 48, (20, 200, 20))
    return [
        red,
        red.copy(),
        red.copy(),
        blue,
        blue.copy(),
        green,
        green.copy(),
    ]


def test_diff_filter_parity_on_scene_changes():
    frames = _scene_frames()
    cli_pairs = _cli().diff_filter(frames)
    gui_pairs = _gui().diff_filter(frames)
    assert len(cli_pairs) == len(gui_pairs)
    for left, right in zip(cli_pairs, gui_pairs):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])


def test_ssim_filter_parity_on_scene_changes():
    frames = _scene_frames()
    pairs = _cli().diff_filter(frames)
    cli_fails = _cli().structural_similarity_filter(pairs)
    gui_fails = _gui().structural_similarity_filter(pairs)
    assert len(cli_fails) == len(gui_fails)
    for left, right in zip(cli_fails, gui_fails):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])


@pytest.mark.parametrize("threshold", [0.10, 0.50, 0.90, 0.99])
def test_diff_threshold_parity(threshold, red, blue, green):
    frames = [red, blue, green, green.copy()]
    cli = Mp4ToPdf("in.mp4", "out.pdf", 1, None, threshold, 0.90, verbose=False)
    gui = Mp4ToPdfWorker(RecordingQueue(), "in.mp4", "out.pdf", 1, threshold, 0.90)
    assert len(cli.diff_filter(frames)) == len(gui.diff_filter(frames))


@pytest.mark.parametrize("threshold", [0.10, 0.50, 0.90, 0.99, 1.01])
def test_ssim_threshold_parity(threshold, red, blue):
    pairs = [[red, blue], [red, red.copy()]]
    cli = Mp4ToPdf("in.mp4", "out.pdf", 1, None, 0.90, threshold, verbose=False)
    gui = Mp4ToPdfWorker(RecordingQueue(), "in.mp4", "out.pdf", 1, 0.90, threshold)
    assert len(cli.structural_similarity_filter(pairs)) == len(gui.structural_similarity_filter(pairs))


def test_pipeline_unique_frames_match(red, blue, green):
    frames = [red, red.copy(), blue, green, green.copy()]
    cli_uniques = [pair[0] for pair in _cli().structural_similarity_filter(_cli().diff_filter(frames))]
    gui_uniques = [pair[0] for pair in _gui().structural_similarity_filter(_gui().diff_filter(frames))]
    assert len(cli_uniques) == len(gui_uniques) == 2
    np.testing.assert_array_equal(cli_uniques[0], gui_uniques[0])
    np.testing.assert_array_equal(cli_uniques[1], gui_uniques[1])
