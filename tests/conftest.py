import pytest

from mp4_to_pdf import Mp4ToPdf
from mp4_to_pdf_gui import Mp4ToPdfWorker
from tests.helpers import RecordingQueue, rgb_frame


@pytest.fixture
def rgb():
    return rgb_frame


@pytest.fixture
def red(rgb):
    return rgb(32, 48, (255, 0, 0))


@pytest.fixture
def blue(rgb):
    return rgb(32, 48, (0, 0, 255))


@pytest.fixture
def green(rgb):
    return rgb(32, 48, (0, 255, 0))


@pytest.fixture
def black(rgb):
    return rgb(32, 48, (0, 0, 0))


@pytest.fixture
def white(rgb):
    return rgb(32, 48, (255, 255, 255))


@pytest.fixture
def cli_converter(tmp_path):
    return Mp4ToPdf(
        infile=str(tmp_path / "in.mp4"),
        out=str(tmp_path / "out.pdf"),
        n_frame=1,
        lim=None,
        diff_threshold=0.90,
        ssim_threshold=0.90,
        verbose=False,
    )


@pytest.fixture
def worker(tmp_path):
    return Mp4ToPdfWorker(
        _queue=RecordingQueue(),
        infile=str(tmp_path / "in.mp4"),
        out=str(tmp_path / "out.pdf"),
        n_frame=1,
        diff_threshold=0.90,
        ssim_threshold=0.90,
    )
