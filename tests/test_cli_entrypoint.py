import argparse
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "mp4_to_pdf.py"


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("--out")
    parser.add_argument("--nframe", type=int, default=24)
    parser.add_argument("--lim", type=int, default=None)
    parser.add_argument("--diff", type=float, default=0.90)
    parser.add_argument("--ssim", type=float, default=0.90)
    parser.add_argument("-v", action="store_true")
    return parser


def test_help_exits_zero():
    with patch.object(sys, "argv", ["mp4_to_pdf.py", "--help"]):
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(CLI), run_name="__main__")
    assert exc.value.code == 0


def test_missing_infile_exits_nonzero():
    with patch.object(sys, "argv", ["mp4_to_pdf.py"]):
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(CLI), run_name="__main__")
    assert exc.value.code != 0


@pytest.mark.parametrize("path", ["video.avi", "video.mkv", "video.txt", "video", "VIDEO.MP4", "file.mp3"])
def test_rejects_non_mp4_suffix(path):
    with patch.object(sys, "argv", ["mp4_to_pdf.py", path]):
        with pytest.raises(Exception, match="Infile must be a mp4"):
            runpy.run_path(str(CLI), run_name="__main__")


@pytest.mark.parametrize(
    "infile,expected",
    [
        ("talk.mp4", "talk.pdf"),
        ("C:/a.mp4", "C:/a.pdf"),
        ("C:/dir.mp4/talk.mp4", "C:/dir/talk.pdf"),
        ("file.mp4.backup.mp4", "file.backup.pdf"),
        ("lecture.mp4", "lecture.pdf"),
    ],
)
def test_default_out_file_replacement(infile, expected):
    assert f"{infile.replace('.mp4', '')}.pdf" == expected


def test_parser_defaults():
    args = _parser().parse_args(["in.mp4"])
    assert args.infile == "in.mp4"
    assert args.out is None
    assert args.nframe == 24
    assert args.lim is None
    assert args.diff == 0.90
    assert args.ssim == 0.90
    assert args.v is False


def test_parser_custom_flags():
    args = _parser().parse_args(
        ["clip.mp4", "--out", "x.pdf", "--nframe", "10", "--lim", "5", "--diff", "0.5", "--ssim", "0.8", "-v"]
    )
    assert args.infile == "clip.mp4"
    assert args.out == "x.pdf"
    assert args.nframe == 10
    assert args.lim == 5
    assert args.diff == 0.5
    assert args.ssim == 0.8
    assert args.v is True


@pytest.mark.parametrize("nframe", ["1", "24", "150", "2000"])
def test_parser_nframe_int(nframe):
    args = _parser().parse_args(["in.mp4", "--nframe", nframe])
    assert args.nframe == int(nframe)


@pytest.mark.parametrize("flag,value", [("--nframe", "abc"), ("--diff", "nope"), ("--ssim", "x"), ("--lim", "no")])
def test_parser_rejects_invalid_types(flag, value):
    with pytest.raises(SystemExit):
        _parser().parse_args(["in.mp4", flag, value])


def test_verbose_flag_default_false():
    parser = _parser()
    assert parser.parse_args(["a.mp4"]).v is False
    assert parser.parse_args(["a.mp4", "-v"]).v is True
