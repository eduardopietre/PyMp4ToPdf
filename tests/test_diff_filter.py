import numpy as np
import pytest

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker
from tests.helpers import RecordingQueue, rgb_frame


def _cli(threshold=0.90, verbose=False):
    return Mp4ToPdf("in.mp4", "out.pdf", 1, None, threshold, 0.90, verbose=verbose)


def _gui(threshold=0.90):
    return Mp4ToPdfWorker(RecordingQueue(), "in.mp4", "out.pdf", 1, threshold, 0.90)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_empty_images_return_no_pairs(factory):
    assert factory().diff_filter([]) == []


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_single_image_return_no_pairs(factory, red):
    assert factory().diff_filter([red]) == []


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_identical_frames_are_not_pairs(factory, red):
    pairs = factory(threshold=0.90).diff_filter([red, red.copy(), red.copy()])
    assert pairs == []


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_different_frames_are_pairs(factory, red, blue):
    pairs = factory(threshold=0.90).diff_filter([red, blue])
    assert len(pairs) == 1
    np.testing.assert_array_equal(pairs[0][0], blue)
    np.testing.assert_array_equal(pairs[0][1], red)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_pair_order_is_current_then_previous(factory, red, blue):
    pair = factory().diff_filter([red, blue])[0]
    np.testing.assert_array_equal(pair[0], blue)
    np.testing.assert_array_equal(pair[1], red)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_multiple_scene_changes(factory, red, blue, green):
    pairs = factory().diff_filter([red, blue, green])
    assert len(pairs) == 2
    np.testing.assert_array_equal(pairs[0][0], blue)
    np.testing.assert_array_equal(pairs[1][0], green)


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_change_then_stable_frames(factory, red, blue):
    frames = [red, red.copy(), blue, blue.copy(), blue.copy()]
    pairs = factory().diff_filter(frames)
    assert len(pairs) == 1
    np.testing.assert_array_equal(pairs[0][0], blue)


@pytest.mark.parametrize("threshold, expect_pair", [(0.01, False), (0.50, False), (0.99, False), (1.01, True)])
def test_identical_frames_need_threshold_above_one_to_match(red, threshold, expect_pair):
    pairs = _cli(threshold=threshold).diff_filter([red, red.copy()])
    assert bool(pairs) is expect_pair


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_threshold_zero_never_emits_pairs(factory, red, blue):
    pairs = factory(threshold=0.0).diff_filter([red, red.copy(), blue])
    assert pairs == []


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_uint8_subtraction_detects_small_pixel_change(factory):
    a = rgb_frame(16, 16, (10, 10, 10))
    b = a.copy()
    b[0, 0] = (11, 10, 10)
    pairs = factory(threshold=0.9999).diff_filter([a, b])
    assert len(pairs) == 1


@pytest.mark.parametrize("factory", [_cli, _gui])
def test_returned_pairs_are_the_original_arrays(factory, red, blue):
    pairs = factory().diff_filter([red, blue])
    assert pairs[0][0] is blue
    assert pairs[0][1] is red


def test_gui_queue_emits_diff_progress_and_completion(red, blue, green):
    worker = _gui()
    worker.diff_filter([red, blue, green])
    codes = [item[0] for item in worker.queue.items]
    assert codes[-1] == Mp4ToPdfWorker.UPDATE_DIFF
    assert worker.queue.items[-1][1] == 1000
    assert Mp4ToPdfWorker.UPDATE_DIFF in codes
    assert all(code == Mp4ToPdfWorker.UPDATE_DIFF for code in codes)


def test_gui_queue_completion_even_when_no_pairs(red):
    worker = _gui()
    worker.diff_filter([red])
    assert worker.queue.items == [(Mp4ToPdfWorker.UPDATE_DIFF, 1000)]


def test_gui_progress_uses_per_mile_scale(red, blue):
    worker = _gui()
    images = [red, blue]
    worker.diff_filter(images)
    mid = worker.queue.items[0]
    assert mid[0] == Mp4ToPdfWorker.UPDATE_DIFF
    assert mid[1] == 1000


@pytest.mark.parametrize("n", [2, 3, 5, 8, 12])
def test_all_different_frames_yield_n_minus_one_pairs(n):
    frames = [rgb_frame(16, 16, (i * 20, 0, 255 - i * 20)) for i in range(n)]
    assert len(_cli().diff_filter(frames)) == n - 1


@pytest.mark.parametrize("n", [2, 4, 7])
def test_all_identical_frames_yield_zero_pairs(n, red):
    frames = [red.copy() for _ in range(n)]
    assert _gui().diff_filter(frames) == []
