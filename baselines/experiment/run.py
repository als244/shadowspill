"""Run selected baseline sweeps, then summarize their results."""

from __future__ import annotations

import argparse
import fcntl
import shutil
import time
from pathlib import Path

from baselines import summary as summaries
from baselines.common import write_json
from baselines.sweep import reporting as sweep_runner

from .commands import report_command, sweep_command
from .config import (
    load_spec,
    matrix_source,
    spec_path_of,
    sweep_matrix,
    variable_values,
)
from .journal import merged_record, now, run_stamp
from .process import run_process, run_sweep
from .reporting import Console, header


def execute(args, spec: dict, selected: list, output: Path, stamp: str, console):
    """Run the selected sweeps, then the summary and the report; the names
    of what failed."""

    started = time.monotonic()
    record_path = output / "runs" / f"{stamp}.json"
    run = {
        "schema_version": 1,
        "run": stamp,
        "spec_path": spec_path_of(args.spec),
        "spec": spec,
        "started_at": now(),
        "finished_at": None,
        "dry_run": args.dry_run,
        "selected": [sweep["name"] for sweep in selected],
        "sweeps": [],
        "report": None,
        "summary": None,
    }
    header(console, spec, args.spec, selected, output)
    if args.dry_run:
        console(
            sweep_runner.field(
                "dry run", "each sweep validates its cases and writes their commands"
            )
        )
    if args.overwrite:
        for sweep in selected:
            held = output / "sweeps" / sweep["name"]
            if held.exists():
                if args.dry_run:
                    console(sweep_runner.field("overwrite", f"would discard {held}"))
                else:
                    shutil.rmtree(held)
                    console(sweep_runner.field("overwrite", f"discarded {held}"))
    failed: list[str] = []
    ending = "interrupted"
    try:
        for index, sweep in enumerate(selected, 1):
            sweep_output = output / "sweeps" / sweep["name"]
            matrix = output / "matrices" / f"{sweep['name']}.json"
            write_json(
                matrix,
                sweep_matrix(
                    matrix_source(sweep), spec["settings"], variable_values(spec)
                ),
            )
            command = sweep_command(
                spec,
                sweep,
                matrix,
                sweep_output,
                limit=args.limit,
                retry_failed=args.retry_failed,
            )
            if args.dry_run:
                # The sweep's own dry run expands and validates every case, so
                # a matrix its engine refuses fails here, not mid-experiment.
                command.append("--dry-run")
            console()
            console(
                sweep_runner.rule(
                    f"Engine {index}/{len(selected)}: {sweep['name']}"
                    f" ({sweep['engine']})"
                )
            )
            began, clock_began, sweep_started = (
                now(),
                sweep_runner.clock(),
                time.monotonic(),
            )
            try:
                returncode = run_sweep(command, sweep_output / "launcher.log", console)
            except KeyboardInterrupt:
                run["sweeps"].append(
                    {
                        **sweep,
                        "output": str(sweep_output),
                        "command": command,
                        "status": "interrupted",
                        "returncode": None,
                        "started_at": began,
                        "finished_at": now(),
                    }
                )
                console(f"ENGINE {sweep['name']}: interrupted")
                raise
            status = (
                "failed"
                if returncode != 0
                else ("planned" if args.dry_run else "completed")
            )
            if status == "failed":
                failed.append(sweep["name"])
            run["sweeps"].append(
                {
                    **sweep,
                    "output": str(sweep_output),
                    "command": command,
                    "status": status,
                    "returncode": returncode,
                    "started_at": began,
                    "finished_at": now(),
                }
            )
            console(
                sweep_runner.field(
                    "engine",
                    f"{status}"
                    + (f" (exit status {returncode})" if returncode else "")
                    + f"; {sweep_runner.duration(time.monotonic() - sweep_started)},"
                    f" {clock_began} to {sweep_runner.clock()}",
                )
            )
            write_json(record_path, run)
            write_json(output / "experiment.json", merged_record(output, spec, run))
        console()
        console(sweep_runner.rule("Report and summary"))
        if not args.skip_report:
            roots = [
                output / "sweeps" / sweep["name"]
                for sweep in spec["sweeps"]
                if args.dry_run or (output / "sweeps" / sweep["name"]).is_dir()
            ]
            if args.dry_run or any(
                next(root.rglob("row.json"), None) is not None for root in roots
            ):
                command = report_command(roots, output / "report")
                returncode = run_process(command, output / "report.log", args.dry_run)
                status = (
                    "planned"
                    if args.dry_run
                    else ("completed" if returncode == 0 else "failed")
                )
                run["report"] = {
                    "output": str(output / "report"),
                    "command": command,
                    "status": status,
                    "returncode": returncode,
                }
                if status == "failed":
                    failed.append("report")
                console(
                    sweep_runner.field(
                        "report", f"{status}: {output / 'report' / 'index.html'}"
                    )
                )
            else:
                console(sweep_runner.field("report", "not drawn: no rows yet"))
            console()
        try:
            summary = summaries.write(output, summaries.summarize(output))
            if args.publish is not None:
                summaries.publish(output, summary, args.publish)
        except (ValueError, OSError, KeyError) as error:
            console(sweep_runner.field("summary", f"failed: {error}"))
            run["summary"] = {"status": "failed", "error": str(error)}
            failed.append("summary")
        else:
            for line in summaries.render(summary).splitlines():
                console(line)
            run["summary"] = {
                "status": "completed",
                "text": str(output / "summary.md"),
                "data": str(output / "summary.json"),
                "published": None if args.publish is None else str(args.publish),
            }
            if args.publish is not None:
                console(sweep_runner.field("published", args.publish / "summary.md"))
        ending = "finished" + (f", failed: {', '.join(failed)}" if failed else "")
    finally:
        run["finished_at"] = now()
        write_json(record_path, run)
        write_json(output / "experiment.json", merged_record(output, spec, run))
        console()
        console(sweep_runner.rule("Experiment " + ending))
        console(
            sweep_runner.field(
                "end",
                f"{sweep_runner.clock()}, after"
                f" {sweep_runner.duration(time.monotonic() - started)}",
            )
        )
        console(sweep_runner.field("record", output / "experiment.json"))
        console(sweep_runner.field("this run", record_path))
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="where the run writes (default: output/ beside the spec)",
    )
    parser.add_argument(
        "--variable",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a value for a variable the spec declares; repeatable",
    )
    parser.add_argument(
        "--engine",
        action="append",
        default=[],
        metavar="NAME",
        help="run the sweeps of this engine; repeatable",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SWEEP",
        help="run this sweep; repeatable",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="attempt the failed, out-of-memory and timed-out cases again",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard what the sweeps run hold and run them from scratch",
    )
    parser.add_argument(
        "--limit", type=int, help="at most this many new attempts per sweep"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="expand and validate every sweep's cases and write their commands;"
        " launch nothing",
    )
    parser.add_argument(
        "--publish",
        type=Path,
        metavar="DIR",
        help="also write the summary and its figures into DIR, the experiment's"
        " results folder",
    )
    parser.add_argument("--skip-report", action="store_true", help="draw no report")
    args = parser.parse_args(argv)
    variables = {}
    for item in args.variable:
        name, equals, value = item.partition("=")
        if not equals or not name or not value:
            parser.error(f"--variable takes NAME=VALUE, not {item!r}")
        variables[name] = value
    try:
        spec = load_spec(args.spec, args.output, variables)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    unknown = set(args.only) - {sweep["name"] for sweep in spec["sweeps"]}
    if unknown:
        parser.error(f"no such sweeps: {sorted(unknown)}")
    unknown = set(args.engine) - {sweep["engine"] for sweep in spec["sweeps"]}
    if unknown:
        parser.error(f"no sweeps of: {sorted(unknown)}")
    selected = [
        sweep
        for sweep in spec["sweeps"]
        if not (args.only or args.engine)
        or sweep["name"] in args.only
        or sweep["engine"] in args.engine
    ]
    output = Path(spec["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another experiment holds this output directory")
        stamp = run_stamp(output)
        console = Console(output / "runs" / f"{stamp}.log")
        try:
            failed = execute(args, spec, selected, output, stamp, console)
        finally:
            console.close()
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
