import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

SSIM_MAX_SIDE = 256
SSIM_MIN_SIDE = 7


def pixel_equal_fraction(current, previous):
    """Fraction of uint8 channel values that are exactly equal.

    For uint8 this matches ``mean(abs(a - b) < 0.01)`` used previously,
    without wrapping subtraction or a temporary float mask.
    """
    return np.count_nonzero(current == previous) / current.size


def downscale_for_ssim(image, max_side=SSIM_MAX_SIDE):
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image
    scale = max_side / largest
    new_width = max(SSIM_MIN_SIDE, int(round(width * scale)))
    new_height = max(SSIM_MIN_SIDE, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def ssim_score(image_a, image_b, confirm_threshold=None, margin=0.03):
    left = downscale_for_ssim(image_a)
    right = downscale_for_ssim(image_b)
    kwargs = {"channel_axis": -1}
    if left.dtype == np.uint8:
        kwargs["data_range"] = 255
    score = structural_similarity(left, right, **kwargs)
    if (
        confirm_threshold is not None
        and left is not image_a
        and abs(score - confirm_threshold) <= margin
    ):
        return structural_similarity(image_a, image_b, **kwargs)
    return score


def _skip_frames(video, amount):
    for _ in range(amount):
        if not video.grab():
            return False
    return True


def iter_sampled_frames(path, n_frame, lim=None, on_open=None, on_progress=None):
    """Yield every ``n_frame``-th decoded frame in OpenCV BGR order.

    Forward skips use grab() rather than seek. On typical MP4 files a seek
    rewinds to the previous keyframe and re-decodes, which is slower than
    demuxing intermediate packets without full decode.
    """
    video = cv2.VideoCapture(path)
    try:
        video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        if on_open is not None:
            on_open(length, fps)

        count = 0
        n_frame = max(n_frame, 1)
        while video.isOpened():
            success, image = video.read()
            if not success:
                break

            frame = image.copy()
            count += n_frame
            if on_progress is not None:
                on_progress(count, length)
            yield frame

            if lim and count > lim:
                break
            if n_frame > 1 and not _skip_frames(video, n_frame - 1):
                break
    finally:
        video.release()


def collect_sampled_frames(path, n_frame, lim=None, on_open=None, on_progress=None):
    return list(
        iter_sampled_frames(
            path,
            n_frame,
            lim=lim,
            on_open=on_open,
            on_progress=on_progress,
        )
    )


def unique_frames_from_iterable(
    frames,
    diff_threshold,
    ssim_threshold,
    on_pair=None,
    on_ssim=None,
):
    """Keep unique slides while only retaining the previous frame plus hits."""
    uniques = []
    previous = None
    image_count = 0
    pair_count = 0

    for frame in frames:
        image_count += 1
        if previous is not None:
            if on_pair is not None:
                on_pair(image_count - 1)
            if pixel_equal_fraction(frame, previous) < diff_threshold:
                pair_count += 1
                if on_ssim is not None:
                    on_ssim(pair_count)
                if ssim_score(frame, previous, confirm_threshold=ssim_threshold) < ssim_threshold:
                    uniques.append(frame)
        previous = frame

    return uniques, image_count, pair_count


def diff_pairs(images, threshold, on_progress=None):
    pairs = []
    total = len(images)
    for index in range(1, total):
        current = images[index]
        previous = images[index - 1]
        if pixel_equal_fraction(current, previous) < threshold:
            pairs.append([current, previous])
        if on_progress is not None:
            on_progress(index + 1, total)
    return pairs


def ssim_filter_pairs(pairs, threshold, on_progress=None):
    fails = []
    total = len(pairs)
    for index, pair in enumerate(pairs):
        if ssim_score(pair[0], pair[1], confirm_threshold=threshold) < threshold:
            fails.append(pair)
        if on_progress is not None:
            on_progress(index + 1, total)
    return fails


def unique_frames_from_pairs(changes):
    return [pair[0] for pair in changes]


def save_bgr_frames_as_pdf(path, images):
    rgb_images = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in images
    ]
    rgb_images[0].save(
        path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=rgb_images[1:],
    )
