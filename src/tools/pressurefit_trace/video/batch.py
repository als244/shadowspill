"""Resumable subprocess rendering and verified publication of a video matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from benchmarking.processes import worker_command

WIDTHS = {"720p": 1280, "1080p": 1920, "1440p": 2560, "4k": 3840}
VIDEOS = ("cuts", "emission-placement", "execution")


def timing(value):
    try:
        values = tuple(float(part) for part in value.split(","))
        if len(values) != 5 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use five positive seconds: explain,motion,build,sweep,hold"
        ) from error


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="Directory containing matrix.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--publish-to",
        type=Path,
        help="Copy each verified movie into this library directory",
    )
    parser.add_argument(
        "--title",
        help="Workload title; individual geometry/ordering/resolution remain visible",
    )
    parser.add_argument("--videos", nargs="+", choices=VIDEOS, default=list(VIDEOS))
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include story frames and detailed examples in all three videos",
    )
    parser.add_argument("--resolution", choices=WIDTHS, default="4k")
    parser.add_argument(
        "--quality", choices=("preview", "high", "best"), default="best"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--motion-fps", type=int, default=30)
    parser.add_argument(
        "--jobs", type=int, default=4, help="Concurrent movie processes"
    )
    parser.add_argument(
        "--emission-timing", type=timing, default=timing("15,12,30,50,8")
    )
    parser.add_argument(
        "--execution-timing", type=timing, default=timing("18,15,30,60,8")
    )
    parser.add_argument(
        "--cut-options-json",
        default="{}",
        help='JSON object of extra cut-render CLI options, e.g. {"explain-seconds":20}',
    )
    return parser


def validate_movie(output, video, *, width, fps, quality):
    from .matrix import file_digest, write

    output = Path(output)
    report = json.loads(
        output.with_suffix(
            ".storyboard.json" if video == "cuts" else ".json"
        ).read_text()
    )
    digest = file_digest(output)
    cache = output.with_suffix(".validation.json")
    if cache.exists():
        previous = json.loads(cache.read_text())
        if previous.get("sha256") == digest and previous.get("decode") == "passed":
            return previous
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = next(
        s for s in json.loads(result.stdout)["streams"] if s["codec_type"] == "video"
    )
    if (
        (stream["width"], stream["height"]) != (width, round(width * 9 / 16))
        or stream["avg_frame_rate"] != f"{fps}/1"
        or int(stream["nb_frames"]) != report["frames"]
        or abs(float(stream["duration"]) - report["seconds"]) > 1 / fps + 0.001
        or report["quality"] != quality
    ):
        raise ValueError(f"encoded movie differs from its render report: {output}")
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(output),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if decoded.stderr.strip():
        raise ValueError(f"decode errors in {output}: {decoded.stderr}")
    review = output.parent / f"{output.stem}_encoded_review"
    review.mkdir(exist_ok=True)
    if video == "cuts":
        shots = report["shots"]
        chosen = [shots[0], shots[-1]]
        chosen += [s for s in shots if s.get("explanation")]
        chosen += [
            next(
                (
                    s
                    for s in shots
                    if s.get("event_index") == cut - 1
                    and s.get("object_journey", {}).get("zoom_progress", 0) >= 0.99
                ),
                shots[-1],
            )
            for cut in report.get("detailed_cuts", [])
        ]
        moments = [
            (s["first_video_frame"] + s["video_frame_count"] / 2) / fps for s in chosen
        ]
    else:
        moments = [
            s["video_start_seconds"] + s["seconds"] / 2
            for s in report["storyboard"]["scenes"]
        ]
    moments = sorted(set(round(t, 3) for t in moments))
    for index, moment in enumerate(moments):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "2",
                "-y",
                "-ss",
                str(moment),
                "-i",
                str(output),
                "-frames:v",
                "1",
                str(review / f"{index + 1:02d}-{moment:07.3f}s.png"),
            ],
            capture_output=True,
            check=True,
        )
    result = {
        "movie": str(output),
        "sha256": digest,
        "decode": "passed",
        "width": width,
        "height": stream["height"],
        "fps": fps,
        "frames": report["frames"],
        "seconds": report["seconds"],
        "quality": quality,
        "review_frames": str(review),
        "review_timestamps_seconds": moments,
        "visual_review": "encoded frames available; not automatically judged",
    }
    write(cache, result)
    return result


def movie_command(args, case, record, video, output):
    common = [
        "--resolution",
        args.resolution,
        "--quality",
        args.quality,
        "--fps",
        str(args.fps),
    ]
    if args.title:
        title = args.title
        if "physical_budget_gib" in record:
            title += f" · {record['physical_budget_gib']:g} GiB"
        common += ["--title", title]
    if args.details:
        common += ["--details"]
    if video == "cuts":
        extra = []
        for key, value in json.loads(args.cut_options_json).items():
            extra.extend(["--" + key.replace("_", "-"), str(value)])
        return worker_command(
            "tools.pressurefit_trace.video",
            [
                "render",
                str(case / record["trace"]),
                "--reduction",
                "answer",
                "--axis",
                "compute",
                *extra,
                *common,
                "--output",
                str(output),
            ],
        )
    options = []
    if record.get("semantics"):
        options += ["--semantics", str(case / record["semantics"])]
    times = (
        args.emission_timing if video == "emission-placement" else args.execution_timing
    )
    for flag, value in zip(
        ("explain", "motion", "build", "sweep", "hold"), times, strict=True
    ):
        options += [f"--{flag}-seconds", str(value)]
    return worker_command(
        "tools.pressurefit_trace.video.plan",
        [
            "render",
            str(case / record["plan"]),
            "--metadata",
            str(case / "metadata.json"),
            "--video",
            video,
            *options,
            *common,
            "--motion-fps",
            str(args.motion_fps),
            "--output",
            str(output),
        ],
    )


def render_batch(args):
    from .matrix import file_digest, write

    if min(args.jobs, args.fps, args.motion_fps) < 1 or args.motion_fps > args.fps:
        raise ValueError(
            "jobs/fps/motion-fps must be positive; motion-fps must not exceed fps"
        )
    if not isinstance(json.loads(args.cut_options_json), dict):
        raise ValueError("cut-options-json must be an object")
    root, output = args.matrix.resolve(), args.output.resolve()
    matrix = json.loads((root / "matrix.json").read_text())
    if not matrix.get("complete"):
        raise ValueError("matrix preparation is not complete")
    output.mkdir(parents=True, exist_ok=True)
    published = args.publish_to.resolve() if args.publish_to else None
    if published:
        published.mkdir(parents=True, exist_ok=True)
        (published / ".gitignore").write_text("*\n")
    # Include source hashes so a resume cannot silently mix renderer versions.
    source = Path(__file__).resolve().parent.parent
    configuration = {
        "matrix_sha256": file_digest(root / "matrix.json"),
        "title": args.title,
        "videos": args.videos,
        "details": args.details,
        "resolution": args.resolution,
        "quality": args.quality,
        "fps": args.fps,
        "motion_fps": args.motion_fps,
        "emission_timing": list(args.emission_timing),
        "execution_timing": list(args.execution_timing),
        "cut_options": json.loads(args.cut_options_json),
        "source_sha256": {
            str(p.relative_to(source)): file_digest(p)
            for p in sorted(source.rglob("*.py"))
        },
    }
    config_path = output / "render_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError(
            "render inputs/options/source changed; use a fresh output directory"
        )
    write(config_path, configuration)
    jobs = [
        (record, video)
        for record in matrix["cases"]
        if record["status"] == "ready"
        for video in args.videos
    ]
    status, lock = {}, threading.Lock()

    def index():
        record = {"total": len(jobs), "movies": list(status.values())}
        write(output / "progress.json", record)
        budgets = any("physical_budget_gib" in item for item in matrix["cases"])
        lines = [
            "# Resolution and ordering videos",
            "",
            "Each feasible combination uses its best PressureFit policy. "
            "Execution timings are simulated.",
            "",
            ("| Budget (GiB) " if budgets else "")
            + "| Microbatch | Ordering | Recompute | Plan | Cut trajectory "
            "| Emission / placement | Execution |",
            ("| --- " if budgets else "")
            + "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in matrix["cases"]:
            cells = [
                str(item.get(k, "")) for k in ("geometry", "ordering", "resolution")
            ]
            if budgets:
                cells.insert(
                    0,
                    f"{item['physical_budget_gib']:g}"
                    if "physical_budget_gib" in item
                    else "—",
                )
            cells.append(
                f"{item['makespan_ns'] / 1e9:.6f} s · {item['candidate_id']}"
                if item.get("makespan_ns") is not None
                else "Infeasible"
            )
            for video in VIDEOS:
                key = f"{item['path']}/{video}"
                value = status.get(key, {})
                if value.get("status") == "complete":
                    label = value["relative"]
                    seconds = round(value["validation"]["seconds"])
                    cells.append(
                        f"[{seconds // 60}:{seconds % 60:02d}]({quote(label)})"
                    )
                else:
                    cells.append(
                        value.get(
                            "status", "queued" if item["status"] == "ready" else "—"
                        )
                    )
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "Infeasible combinations and their candidate failure details "
            "are preserved in matrix.json.",
            "Every completed MP4 passed a full decode and format check. "
            "Encoded review frames are stored beside the source movies.",
            "",
        ]
        (output / "README.md").write_text("\n".join(lines))
        if published:
            shutil.copy2(output / "README.md", published / "README.md")
            write(published / "manifest.json", record)
            shutil.copy2(root / "matrix.json", published / "matrix.json")

    available = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    local = threading.local()
    next_slot = iter(range(args.jobs))

    def movie(item):
        record, video = item
        key = f"{record['path']}/{video}"
        name = "cuts-compute" if video == "cuts" else video
        relative = Path(record["path"]) / f"{name}_{args.resolution}.mp4"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        state_path = destination.with_suffix(".status.json")
        state = {
            "key": key,
            "relative": str(relative),
            "status": "running",
            "started_at": time.time(),
        }
        with lock:
            if not hasattr(local, "slot"):
                local.slot = next(next_slot)
            status[key] = dict(state)
            index()
        try:
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            if (
                previous.get("status") != "complete"
                or not destination.exists()
                or file_digest(destination)
                != previous.get("validation", {}).get("sha256")
            ):
                argv = movie_command(
                    args, root / record["path"], record, video, destination
                )
                cpus = available[local.slot :: args.jobs]
                command = (
                    ["taskset", "--cpu-list", ",".join(map(str, cpus)), *argv]
                    if cpus and shutil.which("taskset")
                    else argv
                )
                state["command"] = command
                write(state_path, state)
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "MPLBACKEND": "Agg",
                }
                with destination.with_suffix(".render.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            else:
                state["command"] = previous.get("command")
            validation = validate_movie(
                destination,
                video,
                width=WIDTHS[args.resolution],
                fps=args.fps,
                quality=args.quality,
            )
            if published:
                target = published / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, target)
                if file_digest(target) != validation["sha256"]:
                    raise ValueError(f"published copy differs: {target}")
                suffix = ".storyboard.json" if video == "cuts" else ".json"
                shutil.copy2(
                    destination.with_suffix(suffix), target.with_suffix(suffix)
                )
            state.update(
                status="complete", validation=validation, finished_at=time.time()
            )
        except Exception as error:
            state.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                finished_at=time.time(),
            )
        write(state_path, state)
        with lock:
            status[key] = dict(state)
            index()
        print(f"{state['status'].upper()} {key}", flush=True)
        return state

    with lock:
        index()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(movie, jobs))
    if any(row["status"] != "complete" for row in results):
        raise RuntimeError(
            "some videos failed; inspect progress.json and rerun to resume"
        )
    write(
        output / "complete.json", {"movies": len(results), "finished_at": time.time()}
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        render_batch(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
