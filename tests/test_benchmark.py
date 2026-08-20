import json
from pathlib import Path

import pytest

import benchmark
from mp4_to_pdf import Mp4ToPdf


def test_default_input_points_at_external_benchmark_file():
    assert benchmark.DEFAULT_INPUT.name == "benchmark_input.mp4"
    assert benchmark.DEFAULT_INPUT.parent.name == "external"
    assert benchmark.DEFAULT_INPUT.parent.parent == Path(__file__).resolve().parents[1]


def test_parse_args_defaults():
    args = benchmark.parse_args([])
    assert args.input == str(benchmark.DEFAULT_INPUT)
    assert args.nframe == 24
    assert args.diff == 0.90
    assert args.ssim == 0.90
    assert args.repeat == 1
    assert args.lim is None
    assert args.out is None
    assert args.json_out is None
    assert args.keep_pdf is False


def test_parse_args_custom(tmp_path):
    json_out = str(tmp_path / "out.json")
    args = benchmark.parse_args(
        [
            "--input",
            "custom.mp4",
            "--out",
            "out.pdf",
            "--nframe",
            "150",
            "--lim",
            "300",
            "--diff",
            "0.5",
            "--ssim",
            "0.8",
            "--repeat",
            "3",
            "--json-out",
            json_out,
            "--keep-pdf",
        ]
    )
    assert args.input == "custom.mp4"
    assert args.out == "out.pdf"
    assert args.nframe == 150
    assert args.lim == 300
    assert args.diff == 0.5
    assert args.ssim == 0.8
    assert args.repeat == 3
    assert args.json_out == json_out
    assert args.keep_pdf is True


@pytest.mark.parametrize(
    "frame_count,n_frame,lim,expected",
    [
        (38861, 24, None, 1 + 38860 // 24),
        (100, 1, None, 100),
        (100, 10, None, 10),
        (100, 10, 20, 3),
        (0, 24, None, 0),
        (50, 0, None, 0),
    ],
)
def test_estimate_sampled_frames(frame_count, n_frame, lim, expected):
    assert benchmark.estimate_sampled_frames(frame_count, n_frame, lim) == expected


def test_estimate_frame_ram_bytes():
    assert benchmark.estimate_frame_ram_bytes(2, 1920, 1080) == 2 * 1920 * 1080 * 3


@pytest.mark.parametrize(
    "size,expected",
    [
        (500, "500.0 B"),
        (2048, "2.0 KB"),
        (1048576, "1.0 MB"),
        (1073741824, "1.0 GB"),
    ],
)
def test_format_bytes(size, expected):
    assert benchmark.format_bytes(size) == expected


def test_probe_video_missing(tmp_path):
    info = benchmark.probe_video(tmp_path / "missing.mp4")
    assert info.exists is False
    assert info.frames == 0


def test_probe_video_reads_metadata(tmp_path):
    from tests.helpers import bgr_frame, write_video

    path = tmp_path / "clip.mp4"
    write_video(path, [bgr_frame(32, 48, (0, 0, 255)) for _ in range(5)], fps=10)
    info = benchmark.probe_video(path)
    assert info.exists is True
    assert info.width == 48
    assert info.height == 32
    assert info.fps == pytest.approx(10.0)
    assert info.frames >= 1
    assert info.size_bytes > 0


def test_run_once_times_each_stage(tmp_path, monkeypatch, red, blue):
    converter = Mp4ToPdf(str(tmp_path / "in.mp4"), str(tmp_path / "out.pdf"), 1, None, 0.90, 0.90)
    monkeypatch.setattr(converter, "get_images", lambda: [red, blue])
    result = benchmark.run_once(converter)
    names = [stage.name for stage in result.stages]
    assert names == ["read", "diff", "ssim", "export"]
    assert result.images == 2
    assert result.pairs == 1
    assert result.uniques == 1
    assert result.total == pytest.approx(sum(stage.seconds for stage in result.stages))
    assert result.export_error == ""
    assert Path(converter.out).is_file()


def test_run_once_records_empty_export(tmp_path, monkeypatch, red):
    converter = Mp4ToPdf(str(tmp_path / "in.mp4"), str(tmp_path / "out.pdf"), 1, None, 0.90, 0.90)
    monkeypatch.setattr(converter, "get_images", lambda: [red, red.copy()])
    result = benchmark.run_once(converter)
    assert result.uniques == 0
    assert result.export_error == "no unique frames to export"
    assert result.stages[-1].seconds == 0.0


def test_summarize_mean_min_max():
    runs = [
        benchmark.RunResult(
            stages=[
                benchmark.StageResult("read", 1.0),
                benchmark.StageResult("diff", 2.0),
                benchmark.StageResult("ssim", 3.0),
                benchmark.StageResult("export", 4.0),
            ],
            total=10.0,
            images=1,
            pairs=1,
            uniques=1,
        ),
        benchmark.RunResult(
            stages=[
                benchmark.StageResult("read", 3.0),
                benchmark.StageResult("diff", 2.0),
                benchmark.StageResult("ssim", 1.0),
                benchmark.StageResult("export", 4.0),
            ],
            total=10.0,
            images=1,
            pairs=1,
            uniques=1,
        ),
    ]
    summary = benchmark.summarize(runs)
    assert summary["runs"] == 2
    assert summary["stages"]["read"]["mean"] == pytest.approx(2.0)
    assert summary["stages"]["read"]["min"] == pytest.approx(1.0)
    assert summary["stages"]["read"]["max"] == pytest.approx(3.0)
    assert summary["total"]["mean"] == pytest.approx(10.0)


def test_format_report_contains_stages_and_input():
    info = benchmark.VideoInfo(path="external/benchmark_input.mp4", exists=True, size_bytes=1000, frames=10, fps=10, width=64, height=48)
    settings = {"nframe": 24, "diff": 0.9, "ssim": 0.9, "lim": None, "sampled": 1, "ram_bytes": 100}
    run = benchmark.RunResult(
        stages=[
            benchmark.StageResult("read", 1.0, "2 images"),
            benchmark.StageResult("diff", 1.0, "1 pairs"),
            benchmark.StageResult("ssim", 2.0, "1 uniques"),
            benchmark.StageResult("export", 1.0, "10 bytes"),
        ],
        total=5.0,
        images=2,
        pairs=1,
        uniques=1,
    )
    report = benchmark.format_report(info, settings, [run], benchmark.summarize([run]))
    assert "PyMp4ToPdf benchmark" in report
    assert "external/benchmark_input.mp4" in report
    assert "read" in report
    assert "ssim" in report
    assert "Summary (seconds)" in report


def test_build_json_roundtrip():
    info = benchmark.VideoInfo(path="in.mp4", exists=True)
    settings = {"nframe": 24, "diff": 0.9, "ssim": 0.9, "lim": None, "sampled": 1, "ram_bytes": 1}
    run = benchmark.RunResult(
        stages=[benchmark.StageResult(name, 0.1) for name in benchmark.STAGE_NAMES],
        total=0.4,
        images=2,
        pairs=1,
        uniques=1,
    )
    payload = benchmark.build_json(info, settings, [run], benchmark.summarize([run]))
    encoded = json.dumps(payload)
    assert "read" in encoded
    assert payload["runs"][0]["images"] == 2


def test_main_missing_input(tmp_path, capsys):
    code = benchmark.main(["--input", str(tmp_path / "missing.mp4")])
    assert code == 2
    err = capsys.readouterr().err
    assert "Benchmark input not found" in err


@pytest.mark.parametrize("flag,value", [("--repeat", "0"), ("--nframe", "0")])
def test_main_rejects_invalid_counts(flag, value, tmp_path, monkeypatch, capsys):
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"not-a-real-video")
    monkeypatch.setattr(
        benchmark,
        "probe_video",
        lambda _path: benchmark.VideoInfo(path=str(fake), exists=True, frames=10, fps=10, width=8, height=8),
    )
    code = benchmark.main(["--input", str(fake), flag, value])
    assert code == 2


def test_main_runs_and_writes_json(tmp_path, monkeypatch, red, blue):
    infile = tmp_path / "clip.mp4"
    infile.write_bytes(b"fake")
    json_out = tmp_path / "bench.json"
    out_pdf = tmp_path / "out.pdf"

    monkeypatch.setattr(
        benchmark,
        "probe_video",
        lambda _path: benchmark.VideoInfo(
            path=str(infile), exists=True, size_bytes=4, frames=2, fps=10, width=32, height=48
        ),
    )

    original_init = Mp4ToPdf.__init__

    def fake_init(self, infile, out, n_frame, lim, diff_threshold, ssim_threshold, verbose=False):
        original_init(self, infile, out, n_frame, lim, diff_threshold, ssim_threshold, verbose)
        self.get_images = lambda: [red, blue]

    monkeypatch.setattr(Mp4ToPdf, "__init__", fake_init)

    code = benchmark.main(
        ["--input", str(infile), "--out", str(out_pdf), "--json-out", str(json_out), "--repeat", "2"]
    )
    assert code == 0
    payload = json.loads(json_out.read_text())
    assert payload["summary"]["runs"] == 2
    assert payload["runs"][0]["images"] == 2
    assert out_pdf.is_file()


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        benchmark.parse_args(["--help"])
    assert exc.value.code == 0
