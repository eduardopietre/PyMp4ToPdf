from pathlib import Path

import cv2
import numpy as np
from pypdf import PdfReader

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker


def rgb_frame(height, width, color):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def bgr_frame(height, width, color):
    return rgb_frame(height, width, color)


def write_video(path, frames_bgr, fps=10):
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def assert_valid_pdf(path, page_count=None):
    path = Path(path)
    assert path.is_file()
    assert path.read_bytes().startswith(b"%PDF")
    reader = PdfReader(str(path))
    if page_count is not None:
        assert len(reader.pages) == page_count
    return reader


class FakeVideoCapture:
    """Mimics cv2.VideoCapture read/seek behavior used by get_images()."""

    def __init__(self, frames, fps=30.0):
        self.frames = frames
        self.fps = fps
        self.idx = 0
        self.opened = True
        self.grab_count = 0
        self.seek_positions = []

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.opened and 0 <= self.idx < len(self.frames):
            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame.copy()
        return False, None

    def grab(self):
        self.grab_count += 1
        if self.opened and 0 <= self.idx < len(self.frames):
            self.idx += 1
            return True
        return False

    def set(self, prop, value):
        if prop == 1 or prop == cv2.CAP_PROP_POS_FRAMES:
            self.seek_positions.append(int(value))
            self.idx = int(value)
            return True
        return False

    def release(self):
        self.opened = False


class RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def empty(self):
        return not self.items

    def get(self):
        return self.items.pop(0)


def make_cli(**kwargs):
    params = dict(
        infile="in.mp4",
        out="out.pdf",
        n_frame=1,
        lim=None,
        diff_threshold=0.90,
        ssim_threshold=0.90,
        verbose=False,
    )
    params.update(kwargs)
    return Mp4ToPdf(**params)


def make_gui(**kwargs):
    params = dict(
        _queue=RecordingQueue(),
        infile="in.mp4",
        out="out.pdf",
        n_frame=1,
        diff_threshold=0.90,
        ssim_threshold=0.90,
    )
    params.update(kwargs)
    return Mp4ToPdfWorker(**params)
