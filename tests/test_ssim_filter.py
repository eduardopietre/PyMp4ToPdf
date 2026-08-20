import numpy as np
import pytest
from skimage.metrics import structural_similarity

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker
from tests.helpers import RecordingQueue, rgb_frame


def _cli(threshold=0.90):
    return Mp4ToPdf("in.mp4", "out.pdf", 1, None, 0.90, threshold, verbose=False)


def _gui(threshold=0.90):
    return Mp4ToPdfWorker(RecordingQueue(), "in.mp4", "out.pdf", 1, 0.90, threshold)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_empty_pairs_return_empty(factory):
    assert factory().structural_similarity_filter([]) == []


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_identical_rgb_frames_are_not_unique(factory, red):
    fails = factory(threshold=0.90).structural_similarity_filter([[red, red.copy()]])
    assert fails == []


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_very_different_rgb_frames_are_unique(factory, red, blue):
    fails = factory(threshold=0.90).structural_similarity_filter([[red, blue]])
    assert len(fails) == 1
    assert fails[0][0] is red
    assert fails[0][1] is blue


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_threshold_one_treats_even_identical_as_not_below(factory, red):
    fails = factory(threshold=1.0).structural_similarity_filter([[red, red.copy()]])
    assert fails == []


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_threshold_above_one_keeps_identical_pairs(factory, red):
    fails = factory(threshold=1.01).structural_similarity_filter([[red, red.copy()]])
    assert len(fails) == 1


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_mixed_pairs_keep_only_dissimilar(factory, red, blue):
    pairs = [[red, red.copy()], [red, blue], [blue, blue.copy()]]
    fails = factory(threshold=0.90).structural_similarity_filter(pairs)
    assert len(fails) == 1
    assert fails[0] is pairs[1]


@pytest.mark.parametrize("size", [(7, 7), (8, 8), (16, 32), (32, 48), (64, 64)])
def test_rgb_frames_do_not_raise_win_size_error(size):
    height, width = size
    a = rgb_frame(height, width, (255, 0, 0))
    b = rgb_frame(height, width, (0, 0, 255))
    result = _gui().structural_similarity_filter([[a, b]])
    assert len(result) == 1


def test_regression_channel_axis_required_for_rgb():
    a = rgb_frame(8, 8, (200, 10, 10))
    b = rgb_frame(8, 8, (10, 10, 200))
    _gui().structural_similarity_filter([[a, b]])
    with pytest.raises(ValueError, match="win_size exceeds image extent"):
        structural_similarity(a, b)


def test_frames_smaller_than_default_window_still_error():
    a = rgb_frame(6, 6, (255, 0, 0))
    b = rgb_frame(6, 6, (0, 0, 255))
    with pytest.raises(ValueError, match="win_size exceeds image extent"):
        _cli().structural_similarity_filter([[a, b]])


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_ssim_of_identical_images_is_one(factory, red):
    score = structural_similarity(red, red.copy(), channel_axis=-1)
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_ssim_of_opposite_colors_is_below_default_threshold(factory, red, blue):
    score = structural_similarity(red, blue, channel_axis=-1)
    assert score < 0.90


def test_gui_queue_emits_smi_progress(red, blue):
    worker = _gui()
    worker.structural_similarity_filter([[red, blue], [red, red.copy()]])
    codes_and_values = worker.queue.items
    assert codes_and_values[0][0] == Mp4ToPdfWorker.UPDATE_SMI
    assert codes_and_values[1][0] == Mp4ToPdfWorker.UPDATE_SMI
    assert codes_and_values[-1] == (Mp4ToPdfWorker.UPDATE_DIFF, 1000)


def test_gui_queue_completion_code_is_update_diff_not_smi():
    worker = _gui()
    worker.structural_similarity_filter([])
    assert worker.queue.items == [(Mp4ToPdfWorker.UPDATE_DIFF, 1000)]


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_all_dissimilar_pairs_are_kept(n, red, blue):
    pairs = [[red, blue] for _ in range(n)]
    assert len(_cli().structural_similarity_filter(pairs)) == n


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_single_channel_difference_can_drop_ssim(channel, red):
    other = red.copy()
    other[:, :, channel] = 255 - other[:, :, channel]
    fails = _gui(threshold=0.99).structural_similarity_filter([[red, other]])
    assert len(fails) == 1
