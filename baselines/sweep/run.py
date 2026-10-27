"Run a matrix of independent, resumable cases of one engine on one node."

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import time
from pathlib import Path

from baselines.common import launch, write_json
from baselines.common.provenance import environment_identity, source_identity

from .config import comparable, expand, keep_names
from .process import run_case
from .reporting import (
    clock,
    duration,
    ended_block,
    field,
    labels,
    report,
    started_block,
    tally,
)
from .results import failure_row, latest, to_attempt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", required=True, help="baselines.<engine>")
    parser.add_argument("--config", required=True, type=Path, help="the matrix")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gpus", type=int, required=True, help="single-node torchrun world size"
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="the trainer's interpreter (default: the environment the engine names)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write configs, commands and the index without launching",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the same manifest without overwriting attempts",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="new attempts for failed, out-of-memory and timed-out cases",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue after a failure that is not an out-of-memory",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=3600,
        help="seconds per attempt, setup and compilation included",
    )
    parser.add_argument(
        "--limit", type=int, help="at most this many new attempts this invocation"
    )
    args = parser.parse_args(argv)
    if (
        args.gpus < 1
        or args.case_timeout <= 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("GPU count, timeout and limit must be positive")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    try:
        module = launch.engine_module(args.engine)
        cases = expand(json.loads(args.config.read_text()), args.gpus, module.Settings)
        cases = keep_names(cases, args.output.resolve())
    except ValueError as error:
        parser.error(str(error))
    if args.python is None:
        args.python = launch.interpreter(module.ENVIRONMENT)
    if not os.access(args.python, os.X_OK):
        parser.error(
            f"{args.python} is not an interpreter: build the {args.engine}"
            " engine's environment or name --python"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another sweep launcher holds this output directory")
        manifest = {
            "schema_version": 1,
            "engine": args.engine,
            "world_size": args.gpus,
            "python": str(args.python.resolve()),
            "source": source_identity(["baselines/sweep", *module.SOURCE_PATHS]),
            "environment": environment_identity(args.python),
            "cases": [
                {"name": name, "settings": settings.as_dict()}
                for name, settings in cases
            ],
        }
        path = output / "manifest.json"
        changes = None
        if path.exists():
            held = json.loads(path.read_text())
            differences = comparable(held, manifest)
            if not args.resume or differences:
                parser.error(
                    "existing output requires --resume with the same engine,"
                    " world size, Python and environment"
                    + (
                        f"; this one differs in {', '.join(differences)}"
                        if differences
                        else ""
                    )
                )
            before = {item["name"] for item in held["cases"]}
            after = {item["name"] for item in manifest["cases"]}
            changes = (len(after - before), len(before - after))
        write_json(path, manifest)
        from baselines.common.preparation import groups, preview

        preparation_groups = groups(
            cases,
            engine=args.engine,
            world_size=args.gpus,
            manifest=manifest,
            output=output,
        )
        unique_preparations = {
            str(group["directory"]): group for group in preparation_groups.values()
        }
        for group in unique_preparations.values():
            preview(group, engine=args.engine, python=args.python, world_size=args.gpus)
        for name, settings in cases:
            case = output / "cases" / name
            write_json(case / "config.json", settings.as_dict())
            first = launch.command(
                args.engine, args.python, args.gpus, settings, case / "attempt-001"
            )
            if name in preparation_groups:
                first += [
                    "--prepared",
                    str(preparation_groups[name]["directory"] / "artifact"),
                ]
            (case / "command.txt").write_text(shlex.join(first) + "\n")
        report(output, cases, args.engine, args.gpus)
        knobs, label_of = labels(cases, args.gpus)
        lines = [
            field("cases", f"{len(cases)} at world size {args.gpus}"),
            field("interpreter", args.python),
            field("varying", ", ".join(knobs) if knobs else "nothing"),
        ]
        if unique_preparations:
            lines.append(
                field(
                    "preparations",
                    f"{len(unique_preparations)} groups; {output / 'preparations'}",
                )
            )
        if changes and any(changes):
            lines.append(
                field(
                    "changed",
                    f"{changes[0]} new cases; {changes[1]} no longer named, kept"
                    " as they ran but not counted",
                )
            )
        lines += [
            field(
                "so far",
                tally(latest(output / "cases" / name)["status"] for name, _ in cases),
            ),
            field("index", output / "README.md"),
            field("cases dir", output / "cases"),
        ]
        if args.dry_run:
            lines.append(
                field("dry run", "every case validated and its command written")
            )
        print("\n".join(lines), flush=True)
        if args.dry_run:
            return
        started, began = time.monotonic(), clock()
        print(field("start", began), flush=True)
        try:
            run_cases(args, cases, output, manifest, label_of)
        finally:
            print(
                "\n".join(
                    [
                        "",
                        f"SWEEP {args.engine}",
                        field(
                            "outcomes",
                            tally(
                                latest(output / "cases" / name)["status"]
                                for name, _ in cases
                            ),
                        ),
                        field(
                            "this run",
                            f"{duration(time.monotonic() - started)},"
                            f" {began} to {clock()}",
                        ),
                    ]
                ),
                flush=True,
            )


def run_cases(args, cases, output, manifest, label_of):
    """Attempt every case that is due, in the matrix's order."""

    width = len(str(len(cases)))
    from baselines.common.preparation import ensure, groups

    preparation_groups = groups(
        cases,
        engine=args.engine,
        world_size=args.gpus,
        manifest=manifest,
        output=output,
    )
    prepared = {}
    attempted = 0
    for index, (name, settings) in enumerate(cases, 1):
        case = output / "cases" / name
        if not to_attempt(latest(case), args.retry_failed):
            continue
        group = preparation_groups.get(name)
        artifact = None
        if group is not None:
            key = str(group["directory"])
            if key not in prepared:
                prepared[key] = ensure(
                    group,
                    engine=args.engine,
                    python=args.python,
                    world_size=args.gpus,
                    timeout=args.case_timeout,
                )
            artifact = prepared[key]
        position = f"[{index:>{width}}/{len(cases)}]"
        attempts = list(case.glob("attempt-*"))
        number = (
            max((int(path.name.split("-")[-1]) for path in attempts), default=0) + 1
        )
        attempt = case / f"attempt-{number:03d}"
        attempt.mkdir()
        cmd = launch.command(args.engine, args.python, args.gpus, settings, attempt)
        if artifact is not None:
            cmd += ["--prepared", str(artifact)]
        (attempt / "command.txt").write_text(shlex.join(cmd) + "\n")
        print(
            "\n".join(
                [
                    "",
                    *started_block(
                        position, name, label_of[name], settings, args.gpus, attempt
                    ),
                ]
            ),
            flush=True,
        )
        try:
            outcome = run_case(cmd, attempt, settings, args.gpus, args.case_timeout)
            if outcome["status"] != "completed":
                write_json(
                    attempt / "row.json",
                    failure_row(
                        attempt,
                        outcome,
                        settings,
                        args.gpus,
                        args.engine,
                        manifest,
                    ),
                )
        finally:
            report(output, cases, args.engine, args.gpus)
        print("\n".join(ended_block(position, name, outcome, attempt)), flush=True)
        attempted += 1
        if (
            outcome["status"] not in ("completed", "cuda_out_of_memory")
            and not args.continue_on_error
        ):
            raise SystemExit(f"Stopped for diagnosis: {attempt / 'console.log'}")
        if args.limit is not None and attempted >= args.limit:
            break
