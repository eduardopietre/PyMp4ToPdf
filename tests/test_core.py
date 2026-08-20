import numpy as np
import pytest

from mp4_to_pdf_core import (
    downscale_for_ssim,
    pixel_equal_fraction,
    ssim_score,
    unique_frames_from_iterable,
)
from tests.helpers import rgb_frame


def test_pixel_equal_fraction_matches_legacy_uint8_formula():
    a = rgb_frame(16, 16, (10, 20, 30))
    b = a.copy()
    b[0, 0] = (11, 20, 30)
    legacy = np.mean(np.abs(a - b) < 0.01)
    assert pixel_equal_fraction(a, b) == pytest.approx(legacy)
    assert pixel_equal_fraction(a, a) == pytest.approx(1.0)


def test_downscale_for_ssim_leaves_small_images_untouched():
    frame = rgb_frame(64, 48, (1, 2, 3))
    assert downscale_for_ssim(frame) is frame


def test_downscale_for_ssim_shrinks_large_images():
    frame = rgb_frame(1080, 1920, (8, 16, 32))
    small = downscale_for_ssim(frame, max_side=256)
    assert max(small.shape[0], small.shape[1]) == 256
    assert small.shape[2] == 3
    assert small.shape[0] < 1080
    assert small.shape[1] < 1920


def test_ssim_score_identical_is_one():
    frame = rgb_frame(32, 48, (200, 10, 10))
    assert ssim_score(frame, frame.copy()) == pytest.approx(1.0)


def test_ssim_score_opposite_colors_below_default_threshold():
    red = rgb_frame(32, 48, (255, 0, 0))
    blue = rgb_frame(32, 48, (0, 0, 255))
    assert ssim_score(red, blue) < 0.90


def test_ssim_score_confirm_uses_full_image_near_threshold():
    large_a = rgb_frame(400, 400, (40, 40, 40))
    large_b = large_a.copy()
    large_b[:40, :40] = (200, 200, 200)
    score = ssim_score(large_a, large_b, confirm_threshold=0.90, margin=1.0)
    assert 0.0 <= score <= 1.0


def test_unique_frames_from_iterable_keeps_new_slides_only():
    red = rgb_frame(32, 48, (255, 0, 0))
    blue = rgb_frame(32, 48, (0, 0, 255))
    green = rgb_frame(32, 48, (0, 255, 0))
    uniques, image_count, pair_count = unique_frames_from_iterable(
        [red, red.copy(), blue, blue.copy(), green],
        diff_threshold=0.90,
        ssim_threshold=0.90,
    )
    assert image_count == 5
    assert pair_count == 2
    assert len(uniques) == 2
    np.testing.assert_array_equal(uniques[0], blue)
    np.testing.assert_array_equal(uniques[1], green)
