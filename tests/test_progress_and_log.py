import datetime

import pytest

from mp4_to_pdf import Mp4ToPdf


def _make(verbose=True):
    return Mp4ToPdf("demo.mp4", "out.pdf", 1, None, 0.90, 0.90, verbose=verbose)


@pytest.mark.parametrize("text", ["hello", "Reading file", "", "unicode ✓"])
def test_log_prints_when_verbose(capsys, text):
    _make(verbose=True).log(text)
    assert capsys.readouterr().out == f"{text}\n"


@pytest.mark.parametrize("text", ["hello", "Reading file", ""])
def test_log_silent_when_not_verbose(capsys, text):
    _make(verbose=False).log(text)
    assert capsys.readouterr().out == ""


def test_progress_bar_silent_when_not_verbose(capsys):
    _make(verbose=False).progress_bar(5, 10)
    assert capsys.readouterr().out == ""


def test_progress_bar_zero_total_raises_when_verbose():
    with pytest.raises(ZeroDivisionError):
        _make(verbose=True).progress_bar(0, 0)


def test_progress_bar_zero_total_ok_when_not_verbose():
    _make(verbose=False).progress_bar(0, 0)


def test_progress_bar_complete_prints_newline(capsys):
    _make(verbose=True).progress_bar(10, 10)
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert "100.0%" in out
    assert "Complete" in out


def test_progress_bar_partial_uses_carriage_return(capsys):
    _make(verbose=True).progress_bar(1, 4)
    out = capsys.readouterr().out
    assert out.startswith("\r")
    assert "25.0%" in out
    assert not out.endswith("Complete\n")


@pytest.mark.parametrize("iteration,total,expected", [
    (0, 10, "0.0%"),
    (5, 10, "50.0%"),
    (10, 10, "100.0%"),
    (1, 3, "33.3%"),
])
def test_progress_bar_percent(capsys, iteration, total, expected):
    _make(verbose=True).progress_bar(iteration, total)
    assert expected in capsys.readouterr().out


def test_progress_bar_custom_prefix_and_fill(capsys):
    _make(verbose=True).progress_bar(2, 4, prefix="Work:", suffix="OK", fill="#", length=10)
    out = capsys.readouterr().out
    assert "Work:" in out
    assert "OK" in out
    assert "#" in out


def test_log_video_info_contents(capsys):
    _make(verbose=True).log_video_info(90, 30)
    out = capsys.readouterr().out
    assert "File demo.mp4:" in out
    assert "FPS: 30." in out
    assert "Lenght: 90 frames." in out
    assert str(datetime.timedelta(seconds=3)) in out


def test_log_video_info_silent_when_not_verbose(capsys):
    _make(verbose=False).log_video_info(90, 30)
    assert capsys.readouterr().out == ""


def test_log_video_info_zero_fps_raises():
    with pytest.raises(ZeroDivisionError):
        _make(verbose=True).log_video_info(10, 0)
