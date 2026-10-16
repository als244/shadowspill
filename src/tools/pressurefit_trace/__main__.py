"""Command line for reusable PressureFit capture, analysis, and animation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect import add_commands
from .data import Trajectory, analyze


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def cut_numbers(value):
    try:
        cuts = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated cut numbers, e.g. 1,15,200"
        ) from error
    if not cuts or cuts[0] < 1:
        raise argparse.ArgumentTypeError("cut numbers start at 1")
    return cuts


def reduction_index(data, value):
    result = data.trace.answer_reduction if value == "answer" else int(value)
    if result is None:
        raise ValueError(
            "this candidate has no answer; choose a numeric reduction index"
        )
    if not 0 <= result < len(data.trace.reductions):
        raise ValueError("reduction index is out of range")
    return result


def output_width(args):
    return (
        args.width
        or {"720p": 1280, "1080p": 1920, "1440p": 2560, "4k": 3840}[args.resolution]
    )


def run_analysis(args):
    data = Trajectory.load(args.input, args.candidate)
    report = analyze(data, args.output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "candidate_index",
                    "compute_seconds",
                    "row_filter",
                    "reductions",
                )
            },
            indent=2,
        )
    )


def run_render(args):
    from .render import render_movie

    data = Trajectory.load(args.input, args.candidate)
    data.select_axis(args.axis)
    data.trace.verify()
    reductions = (
        list(range(len(data.trace.reductions)))
        if args.reduction == "all"
        else [reduction_index(data, args.reduction)]
    )
    result = render_movie(
        data,
        args.output,
        reductions=reductions,
        width=output_width(args),
        fps=args.fps,
        context=args.context_tasks,
        focus_rows=args.focus_rows,
        opening_cuts=args.opening_cuts,
        slow_seconds=args.slow_seconds,
        accelerated_seconds=args.accelerated_seconds,
        pacing=args.pacing,
        explain_cuts=args.explain_cuts,
        explain_seconds=args.explain_seconds,
        detail_at=args.detail_at,
        step_seconds=args.step_seconds,
        bridge_seconds=args.bridge_seconds,
        bridge_hold_seconds=args.bridge_hold_seconds,
        accelerated_hold_seconds=args.accelerated_hold_seconds,
        seed_seconds=args.seed_seconds,
        final_seconds=args.final_seconds,
        cut_zoom=args.cut_zoom,
        zoom_seconds=args.zoom_seconds,
        zoom_hold_seconds=args.zoom_hold_seconds,
        quality=args.quality,
        pressure_scale=args.pressure_scale,
        pressure_view=args.pressure_view,
        focus_mode=args.focus_mode,
    )
    print(f"WROTE {args.output} ({result['seconds']:.2f} s, {result['frames']} frames)")


def run_frame(args):
    from .render import render_frame

    data = Trajectory.load(args.input, args.candidate)
    data.select_axis(args.axis)
    data.trace.verify()
    rid = reduction_index(data, args.reduction)
    cut = len(data.trace.reductions[rid].cuts) if args.cut == "final" else int(args.cut)
    if args.boundary is not None and not 0 <= args.boundary < len(data.edges):
        raise ValueError("boundary position is out of range")
    render_frame(
        data,
        args.output,
        reduction=rid,
        cut=cut,
        width=output_width(args),
        context=args.context_tasks,
        focus_rows=args.focus_rows,
        boundary=args.boundary,
        explain=args.explain,
        journey=True if args.object_timeline else None,
        pressure_scale=args.pressure_scale,
        pressure_view=args.pressure_view,
        focus_mode=args.focus_mode,
    )
    print(f"WROTE {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    add_commands(modes)
    analysis = modes.add_parser(
        "analyze",
        help="Verify a candidate and export every cut, labels, and size distribution",
    )
    frame = modes.add_parser("frame", help="Render one exact state to PNG")
    render = modes.add_parser(
        "render", help="Render seed-to-fit reductions to MP4 with a replay storyboard"
    )
    for command in (analysis, frame, render):
        command.add_argument(
            "input", type=Path, help="Raw trace directory containing manifest.json"
        )
        command.add_argument(
            "--candidate",
            default="winner",
            help="winner or zero-based manifest candidate index",
        )
        command.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Output directory (analyze), PNG path (frame), or MP4 path (render)",
        )
    for command in (frame, render):
        command.add_argument(
            "--focus-mode",
            choices=("cut", "boundary"),
            default="cut",
            help="cut: span the chosen object's prior/next uses; boundary: local tasks",
        )
        command.add_argument(
            "--pressure-view",
            choices=("total", "reducer"),
            default="total",
            help="total: include fixed resident reservations by role; "
            "reducer: omit that pool; both show workspace beneath a flat budget",
        )
        command.add_argument(
            "--pressure-scale",
            choices=("adaptive", "fixed"),
            default="adaptive",
            help="adaptive: zoom toward budget as pressure falls; "
            "fixed: one y-axis scale",
        )
        command.add_argument(
            "--axis",
            choices=("compute", "task"),
            default="compute",
            help="compute: profiled duration widths; "
            "task: equal widths and task indices",
        )
        command.add_argument(
            "--reduction",
            default="all" if command is render else "answer",
            help="Zero-based index, answer, or all (render only)",
        )
        dimensions = command.add_mutually_exclusive_group()
        dimensions.add_argument(
            "--resolution",
            choices=("720p", "1080p", "1440p", "4k"),
            default="1080p",
            help="16:9 output resolution (default: 1080p)",
        )
        dimensions.add_argument(
            "--width",
            type=positive_int,
            help="Custom image width; height is 9/16 of width",
        )
        command.add_argument(
            "--context-tasks",
            type=positive_int,
            default=3,
            help="Tasks on each side of the focused boundary",
        )
        command.add_argument(
            "--focus-rows",
            type=positive_int,
            default=7,
            help="Maximum labeled rows in the local view",
        )
    frame.add_argument(
        "--cut",
        default="final",
        help="Number of applied cuts: 0 is seed; final is last state",
    )
    frame.add_argument(
        "--boundary",
        type=int,
        help="Override focus with this zero-based boundary position",
    )
    frame.add_argument(
        "--explain",
        action="store_true",
        help="Explain the next cut from the displayed state",
    )
    frame.add_argument(
        "--object-timeline",
        action="store_true",
        help="Show the selected object's prior use, removed residency, and next use",
    )
    render.add_argument("--fps", type=positive_int, default=30)
    render.add_argument(
        "--quality",
        choices=("preview", "high", "best"),
        default="high",
        help="H.264 quality: preview=CRF26, high=CRF18, best=CRF14",
    )
    render.add_argument(
        "--explain-cuts",
        type=int,
        default=3,
        help="Spaced detailed examples in the first reduction (0 disables)",
    )
    render.add_argument(
        "--explain-seconds",
        type=positive_float,
        default=15,
        help="Seconds to hold each explanation card",
    )
    render.add_argument(
        "--opening-cuts",
        type=positive_int,
        default=6,
        help="Initial individual cuts; slow holds when explanations are disabled",
    )
    render.add_argument(
        "--slow-seconds",
        type=positive_float,
        default=8,
        help="Seconds to select/apply a detailed cut, excluding explanations",
    )
    render.add_argument(
        "--accelerated-seconds",
        type=positive_float,
        default=45,
        help="Final accelerated segment length per reduction",
    )
    render.add_argument(
        "--detail-at",
        type=cut_numbers,
        help="1-based detailed cuts in the first rendered reduction; "
        "overrides --explain-cuts",
    )
    for flag, default, help_text in (
        ("step-seconds", 1, "Seconds per individual cut between early explanations"),
        (
            "bridge-seconds",
            24,
            "Duration of each longer passage between detailed examples",
        ),
        (
            "bridge-hold-seconds",
            0.6,
            "Minimum time to hold a displayed batch between detailed examples",
        ),
        (
            "accelerated-hold-seconds",
            0.1,
            "Minimum time to hold a batch in the final accelerated passage",
        ),
        ("seed-seconds", 5, "Hold at the start of each reduction"),
        ("final-seconds", 4, "Hold at the final state of each reduction"),
        ("zoom-seconds", 1.2, "Duration of each outward / returning cut camera zoom"),
        (
            "zoom-hold-seconds",
            3,
            "Extra holds split between the uncut full gap and the local result",
        ),
    ):
        render.add_argument(
            f"--{flag}", type=positive_float, default=default, help=help_text
        )
    render.add_argument(
        "--cut-zoom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Animate detailed cuts out to prior/next uses, then back to the boundary",
    )
    render.add_argument(
        "--pacing",
        choices=("overview", "every-cut"),
        default="overview",
        help="overview batches later cuts; every-cut shows each later decision "
        "for one frame",
    )
    analysis.set_defaults(run=run_analysis)
    frame.set_defaults(run=run_frame)
    render.set_defaults(run=run_render)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "dump" and args.quickstart[:1] == ["--"]:
        args.quickstart = args.quickstart[1:]
    try:
        args.run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
