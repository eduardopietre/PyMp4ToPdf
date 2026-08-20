import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from mp4_to_pdf import Mp4ToPdf

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "external" / "benchmark_input.mp4"
STAGE_NAMES = ("read", "diff", "ssim", "export")


@dataclass
class StageResult:
    name: str
    seconds: float
    detail: str = ""


@dataclass
class RunResult:
    stages: list
    total: float
    images: int
    pairs: int
    uniques: int
    output_bytes: int = 0
    export_error: str = ""


@dataclass
class VideoInfo:
    path: str
    exists: bool
    size_bytes: int = 0
    frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0


def timed(func):
    started = time.perf_counter()
    value = func()
    return time.perf_counter() - started, value


def probe_video(path):
    path = Path(path)
    info = VideoInfo(path=str(path), exists=path.is_file())
    if not info.exists:
        return info
    info.size_bytes = path.stat().st_size
    video = cv2.VideoCapture(str(path))
    try:
        info.frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        info.fps = float(video.get(cv2.CAP_PROP_FPS))
        info.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        info.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        video.release()
    return info


def estimate_sampled_frames(frame_count, n_frame, lim):
    if frame_count <= 0 or n_frame < 1:
        return 0
    sampled = 1 + (max(frame_count - 1, 0) // n_frame)
    if lim:
        sampled = min(sampled, 1 + (lim // n_frame))
    return sampled


def estimate_frame_ram_bytes(sampled, width, height):
    return sampled * width * height * 3


def run_once(converter):
    stages = []

    elapsed, images = timed(converter.get_images)
    stages.append(StageResult("read", elapsed, f"{len(images)} images"))

    elapsed, pairs = timed(lambda: converter.diff_filter(images))
    stages.append(StageResult("diff", elapsed, f"{len(pairs)} pairs"))

    elapsed, changes = timed(lambda: converter.structural_similarity_filter(pairs))
    stages.append(StageResult("ssim", elapsed, f"{len(changes)} uniques"))

    uniques = [pair[0] for pair in changes]
    export_error = ""
    output_bytes = 0
    try:
        elapsed, _ = timed(lambda: converter.save_as_pdf(uniques))
        if converter.out and Path(converter.out).is_file():
            output_bytes = Path(converter.out).stat().st_size
    except IndexError:
        elapsed = 0.0
        export_error = "no unique frames to export"
    stages.append(StageResult("export", elapsed, export_error or f"{output_bytes} bytes"))

    total = sum(stage.seconds for stage in stages)
    return RunResult(
        stages=stages,
        total=total,
        images=len(images),
        pairs=len(pairs),
        uniques=len(uniques),
        output_bytes=output_bytes,
        export_error=export_error,
    )


def summarize(runs):
    by_stage = {name: [] for name in STAGE_NAMES}
    totals = []
    for run in runs:
        totals.append(run.total)
        for stage in run.stages:
            by_stage[stage.name].append(stage.seconds)

    summary = {
        "runs": len(runs),
        "total": _stats(totals),
        "stages": {name: _stats(values) for name, values in by_stage.items()},
    }
    return summary


def _stats(values):
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def format_bytes(size):
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def format_report(info, settings, runs, summary):
    lines = [
        "PyMp4ToPdf benchmark",
        f"Input: {info.path}",
        (
            f"Video: {info.width}x{info.height}, {info.frames} frames, "
            f"{info.fps:.2f} fps, {format_bytes(info.size_bytes)}"
        ),
        (
            f"Settings: nframe={settings['nframe']}, diff={settings['diff']}, "
            f"ssim={settings['ssim']}, lim={settings['lim']}"
        ),
        (
            f"Estimate: ~{settings['sampled']} sampled frames, "
            f"~{format_bytes(settings['ram_bytes'])} for decoded RGB"
        ),
        f"Repeats: {len(runs)}",
        "",
    ]

    for index, run in enumerate(runs, start=1):
        lines.append(f"Run {index}")
        for stage in run.stages:
            share = (stage.seconds / run.total * 100) if run.total else 0.0
            lines.append(
                f"  {stage.name:<8} {stage.seconds:8.3f}s  ({share:5.1f}%)  {stage.detail}"
            )
        lines.append(f"  {'total':<8} {run.total:8.3f}s")
        lines.append("")

    lines.append("Summary (seconds)")
    lines.append(f"  {'stage':<8} {'mean':>8} {'min':>8} {'max':>8} {'stdev':>8}")
    for name in STAGE_NAMES:
        stats = summary["stages"][name]
        lines.append(
            f"  {name:<8} {stats['mean']:8.3f} {stats['min']:8.3f} "
            f"{stats['max']:8.3f} {stats['stdev']:8.3f}"
        )
    total = summary["total"]
    lines.append(
        f"  {'total':<8} {total['mean']:8.3f} {total['min']:8.3f} "
        f"{total['max']:8.3f} {total['stdev']:8.3f}"
    )
    return "\n".join(lines)


def build_json(info, settings, runs, summary):
    return {
        "input": asdict(info),
        "settings": settings,
        "runs": [
            {
                "total": run.total,
                "images": run.images,
                "pairs": run.pairs,
                "uniques": run.uniques,
                "output_bytes": run.output_bytes,
                "export_error": run.export_error,
                "stages": [asdict(stage) for stage in run.stages],
            }
            for run in runs
        ],
        "summary": summary,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark MP4 to PDF conversion stages.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the .mp4 input. Defaults to external/benchmark_input.mp4.",
    )
    parser.add_argument("--out", help="PDF output path. Defaults to a temporary file.")
    parser.add_argument("--nframe", type=int, default=24, help="Read every N'th frame. Defaults to 24.")
    parser.add_argument("--lim", type=int, default=None, help="Stop after this frame index.")
    parser.add_argument("--diff", type=float, default=0.90, help="Difference threshold. Defaults to 0.90.")
    parser.add_argument("--ssim", type=float, default=0.90, help="SSIM threshold. Defaults to 0.90.")
    parser.add_argument("--repeat", type=int, default=1, help="How many timed runs. Defaults to 1.")
    parser.add_argument("--json-out", help="Write machine-readable results to this JSON file.")
    parser.add_argument("--keep-pdf", action="store_true", help="Keep the PDF when using a temporary output.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    info = probe_video(args.input)
    if not info.exists:
        print(f"Benchmark input not found: {args.input}", file=sys.stderr)
        print("Place the video at external/benchmark_input.mp4 or pass --input.", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be at least 1.", file=sys.stderr)
        return 2
    if args.nframe < 1:
        print("--nframe must be at least 1.", file=sys.stderr)
        return 2

    sampled = estimate_sampled_frames(info.frames, args.nframe, args.lim)
    ram_bytes = estimate_frame_ram_bytes(sampled, info.width, info.height)
    settings = {
        "nframe": args.nframe,
        "diff": args.diff,
        "ssim": args.ssim,
        "lim": args.lim,
        "sampled": sampled,
        "ram_bytes": ram_bytes,
    }

    tmp_dir = None
    out_file = args.out
    if not out_file:
        tmp_dir = tempfile.TemporaryDirectory(prefix="pymp4topdf-bench-")
        out_file = str(Path(tmp_dir.name) / "benchmark_output.pdf")

    runs = []
    try:
        for _ in range(args.repeat):
            converter = Mp4ToPdf(
                args.input,
                out_file,
                args.nframe,
                args.lim,
                args.diff,
                args.ssim,
                verbose=False,
            )
            runs.append(run_once(converter))
    finally:
        if tmp_dir is not None and not args.keep_pdf:
            tmp_dir.cleanup()

    summary = summarize(runs)
    print(format_report(info, settings, runs, summary))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(build_json(info, settings, runs, summary), indent=2))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
