"""Render saved simulated plans as emission/placement and execution videos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .data import PlanData
from .placement import PlacementReplay
from .storyboard import Timing, build_story, state_at


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def indices(value):
    try:
        result = [int(v.strip()) - 1 for v in value.split(",")]
        if min(result) < 0:
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated numbers starting at 1"
        ) from error
    return list(dict.fromkeys(result))


def parser():
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("analyze", "Verify placement and export action/allocation evidence"),
        ("frame", "Render one scene state to PNG"),
        ("gallery", "Render a review still for every scene"),
        ("render", "Render a teaching video to MP4"),
    ):
        command = commands.add_parser(
            name,
            help=help_text,
            description=help_text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        command.add_argument("plan", type=Path, help="Saved AnnotatedProgramPlan JSON")
        command.add_argument(
            "--semantics", type=Path, help="Semantic labels JSON for this program"
        )
        command.add_argument(
            "--metadata",
            type=Path,
            help="Optional winner/context JSON (ordering, geometry, resolution)",
        )
        command.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Output directory (analyze/gallery), PNG (frame), or MP4 (render)",
        )
        if name == "analyze":
            continue
        command.add_argument(
            "--video",
            choices=("emission-placement", "execution"),
            default="emission-placement",
            help="Which video to compose",
        )
        command.add_argument(
            "--chapter",
            choices=("all", "emission", "placement"),
            default="all",
            help="Chapters to keep in the emission-placement video",
        )
        command.add_argument("--title", help="Workload label in the header")
        command.add_argument(
            "--resolution",
            choices=("720p", "1080p", "1440p", "4k"),
            default="4k",
            help="16:9 output dimensions; 4k is 3840 x 2160",
        )
        command.add_argument(
            "--window-tasks",
            type=positive_int,
            default=8,
            help="Approximate tasks in the sliding focus window",
        )
        command.add_argument(
            "--actions",
            type=indices,
            help="Emission examples: 1-based schedule actions, in presentation order",
        )
        command.add_argument(
            "--tasks",
            type=indices,
            help="Execution examples: 1-based selected task positions with waits",
        )
        command.add_argument(
            "--placements",
            type=indices,
            help="Detailed examples: 1-based placement-order numbers",
        )
        for flag, default, description in (
            ("explain-seconds", 12, "Hold each detailed explanation"),
            (
                "motion-seconds",
                10,
                "Follow an emission example; execution examples use twice this",
            ),
            ("build-seconds", 30, "Final accelerated placement construction"),
            ("sweep-seconds", 40, "Sliding sweep across the whole program"),
            ("hold-seconds", 6, "Introduction, transition, and final holds"),
        ):
            command.add_argument(
                f"--{flag}", type=positive, default=default, help=description
            )
        if name == "frame":
            command.add_argument(
                "--scene", type=positive_int, default=1, help="1-based storyboard scene"
            )
            command.add_argument(
                "--at", type=float, default=0.5, help="Fraction through the scene, 0..1"
            )
        if name == "render":
            command.add_argument(
                "--fps", type=positive_int, default=30, help="Output frames per second"
            )
            command.add_argument(
                "--motion-fps",
                type=positive_int,
                default=15,
                help="Distinct animation states per second, repeated into output FPS",
            )
            command.add_argument(
                "--quality",
                choices=("preview", "high", "best"),
                default="best",
                help="H.264 CRF: preview 26, high 18, best 14; independent of pixels",
            )
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(args):
    data = PlanData.load(args.plan, semantics=args.semantics, metadata=args.metadata)
    replay = PlacementReplay(data)
    verified = replay.verify()
    print(
        f"Verified {verified['verified_offsets']:,} allocation offsets; "
        f"{len(data.actions):,} actions",
        flush=True,
    )
    if args.command == "analyze":
        args.output.mkdir(parents=True, exist_ok=True)
        write(
            args.output / "analysis.json",
            {**data.summary(), "placement_replay": verified},
        )
        with (args.output / "actions.jsonl").open("w") as file:
            for action in data.actions:
                file.write(json.dumps(data.action_detail(action["index"])) + "\n")
        with (args.output / "task_waits.jsonl").open("w") as file:
            for index in range(len(data.tasks)):
                file.write(json.dumps(data.task_detail(index)) + "\n")
        with (args.output / "placement_order.jsonl").open("w") as file:
            for rank, i in enumerate(replay.order):
                file.write(
                    json.dumps(
                        {
                            **data.leases[i],
                            "placement_number": rank + 1,
                            "resident_slice": bool(replay.resident[i]),
                        }
                    )
                    + "\n"
                )
        print(f"WROTE {args.output}", flush=True)
        return
    if args.actions and max(args.actions) >= len(data.actions):
        raise ValueError("action number is outside the schedule")
    if args.tasks and max(args.tasks) >= len(data.tasks):
        raise ValueError("task number is outside the selected program")
    if args.actions and args.video == "execution":
        raise ValueError(
            "use --tasks to choose execution examples; --actions is for emission"
        )
    if args.tasks and args.video != "execution":
        raise ValueError("--tasks applies to the execution video")
    if args.placements and max(args.placements) >= len(data.leases):
        raise ValueError("placement number is outside the layout")
    placements = (
        None
        if args.placements is None
        else [int(replay.order[i]) for i in sorted(args.placements)]
    )
    timing = Timing(
        args.explain_seconds,
        args.motion_seconds,
        args.build_seconds,
        args.sweep_seconds,
        args.hold_seconds,
    )
    story = build_story(
        data,
        replay,
        args.video,
        timing=timing,
        actions=args.actions,
        placements=placements,
        tasks=args.tasks,
    )
    if args.chapter != "all":
        if args.video == "execution":
            raise ValueError("--chapter applies to the emission-placement video")
        story["scenes"] = [
            s for s in story["scenes"] if s["kind"].startswith(args.chapter)
        ]
        elapsed = 0
        for scene in story["scenes"]:
            scene["video_start_seconds"] = elapsed
            elapsed += scene["seconds"]
        story["seconds"] = elapsed
    story.update(
        program_digest=data.program_digest,
        schedule_digest=data.schedule_digest,
        layout_digest=data.layout_digest,
        placement_replay=verified,
    )
    from .render import create_renderer, render_movie

    width = {"720p": 1280, "1080p": 1920, "1440p": 2560, "4k": 3840}[args.resolution]
    renderer = create_renderer(args.video, data, replay, width=width, title=args.title)
    try:
        if args.command == "frame":
            if not 1 <= args.scene <= len(story["scenes"]) or not 0 <= args.at <= 1:
                raise ValueError("invalid scene or --at fraction")
            state = state_at(
                story["scenes"][args.scene - 1],
                args.at,
                data,
                window_tasks=args.window_tasks,
            )
            renderer.draw(state)
            renderer.save(args.output)
            write(
                args.output.with_suffix(".json"), {"state": state, "storyboard": story}
            )
        elif args.command == "gallery":
            for index, scene in enumerate(story["scenes"]):
                state = state_at(scene, 0.5, data, window_tasks=args.window_tasks)
                renderer.draw(state)
                renderer.save(args.output / f"{index + 1:02d}-{scene['kind']}.png")
            write(args.output / "storyboard.json", story)
        else:
            result = render_movie(
                renderer,
                story,
                args.output,
                fps=args.fps,
                quality=args.quality,
                motion_fps=args.motion_fps,
                window_tasks=args.window_tasks,
            )
            print(
                f"WROTE {args.output} ({result['seconds']:.1f}s; "
                f"{result['width']}x{result['height']})",
                flush=True,
            )
    finally:
        renderer.close()


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
