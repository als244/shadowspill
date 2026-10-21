"""Run an experiment: the sweeps of several engines, in order, on one node,
then a summary and a report over everything the experiment holds.

    python -m baselines.experiment --spec experiment.json [--output DIR] \\
        [--variable NAME=VALUE ...] [--engine NAME ...] [--only SWEEP ...] \\
        [--retry-failed] [--overwrite] [--limit N] [--publish DIR] [--dry-run] \\
        [--skip-report]

A spec names the GPU count, the interpreter each engine runs in, the
settings every case of every sweep takes, the sweeps (each an engine, its
matrix, a path from the spec's own directory or the repository's root, and
optionally its own interpreter, timeout and name), and optionally the
GPU-memory budgets its summary tabulates, ``summary_budgets_gib``, and the
variables its settings and matrices refer to. The settings are what the
engines of one experiment share, the model, the geometry, the precision
contract and the run's length; a sweep's matrix names only its engine's
own settings, and one that names a shared setting is refused, so no engine
runs another model, geometry or contract than the rest. A variable, such as
the directory a checkpoint is read from, is declared by name with what its
value is, and a string refers to it as ``{name}``; its value is given at
launch, ``--variable name=VALUE``, or taken from the experiment's record,
and a run without one is refused. Each sweep's matrix, with the settings in
its defaults and every variable replaced, is written to
``<output>/matrices/<name>.json``.

Every sweep runs through ``baselines.sweep`` as its own process, with
``--resume`` once its manifest exists and ``--continue-on-error``, into
``<output>/sweeps/<name>``. Its console goes to ``launcher.log`` beside its
cases and, under the sweep's heading, to the experiment's: what the sweep
holds and what its matrix varies, a line as each case starts, with its
settings, and a line as it ends, with its outcome, throughput, update time,
GPU and host peaks and duration. A case's own output stays in its attempt's
``console.log``. ``--engine`` and ``--only`` run some of the sweeps, by
engine or by name, and the others keep what they hold, so an experiment is
collected one engine at a time or whole; either way it resumes where it
stopped, since a sweep keeps its completed cases, attempts those never
attempted or interrupted, and with ``--retry-failed`` attempts the failed,
out-of-memory and timed-out ones again. ``--overwrite`` discards what the
sweeps it runs hold and runs them from scratch; the other sweeps and the
records of earlier runs stay.

Everything a run writes goes under the experiment's output: ``--output``,
else the spec's ``output``, a path from the spec's folder, else ``output/``
beside the spec, which git ignores, so an experiment keeps its runs and logs
in its own folder. Each run's record and console are
``<output>/runs/<stamp>.json`` and
``.log``; ``experiment.json`` holds the latest spec with its variables'
values, each sweep's latest run and every run. A run ends with the report
of ``baselines.report`` over every sweep the experiment holds, in
``<output>/report`` and drawn against the GPU memory each engine held
(``--memory reserved``), and the summary of ``baselines.summary``, printed
and written with the report's figures to ``summary.md`` and
``summary.json``; ``--publish DIR`` also writes the summary and its figures
into DIR, the experiment's ``results`` folder, with no path of the machine
it ran on.

An interpreter is a path, or the name of an environment beside the running
one: ``baselines`` is ``<prefix>/../baselines/bin/python``.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from baselines import summary as summaries
from baselines import sweep as sweep_runner
from baselines.common import write_json
from baselines.common.launch import engine_module, interpreter
from baselines.common.provenance import ROOT

SPEC_KEYS = {
    "schema_version",
    "output",
    "gpus",
    "python",
    "case_timeout",
    "settings",
    "sweeps",
    "summary_budgets_gib",
    "variables",
}
SWEEP_KEYS = {"name", "engine", "config", "python", "case_timeout"}


#: A variable's name, and a reference to one in a string.
VARIABLE_NAME = re.compile(r"[a-z][a-z0-9_]*")
VARIABLE = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def substitute(value, values: dict[str, str]):
    """``value`` with every ``{name}`` its strings hold replaced by the
    variable's value; a reference to an undeclared variable is refused."""

    if isinstance(value, str):

        def replace(match):
            if match.group(1) not in values:
                raise ValueError(
                    f"{match.group(0)} is not a variable the spec declares"
                )
            return values[match.group(1)]

        return VARIABLE.sub(replace, value)
    if isinstance(value, list):
        return [substitute(item, values) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, values) for key, item in value.items()}
    return value


def variable_values(spec: dict) -> dict[str, str]:
    """The values a loaded spec's variables were given."""

    return {name: entry["value"] for name, entry in spec.get("variables", {}).items()}


def recorded_values(output: Path | None) -> dict[str, str]:
    """The variables' values an experiment's record holds, from its last run."""

    record = None if output is None else Path(output) / "experiment.json"
    if record is None or not record.is_file():
        return {}
    return variable_values(json.loads(record.read_text()).get("spec", {}))


def resolve_config(config: str, base: Path) -> str:
    """A matrix's path, from the spec's directory or else the repository's
    root, as the repository names it when it is inside."""

    path = Path(config)
    candidates = [path] if path.is_absolute() else [base / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            try:
                return str(resolved.relative_to(ROOT))
            except ValueError:
                return str(resolved)
    raise ValueError(f"no matrix at {config}")


def matrix_source(sweep: dict) -> Path:
    """The sweep's matrix file, from the repository root or as given."""

    path = ROOT / sweep["config"]
    return path if path.is_file() else Path(sweep["config"])


def sweep_matrix(path: Path, settings: dict, values: dict | None = None) -> dict:
    """The sweep's matrix with the experiment's settings in its defaults and
    the variables replaced; a matrix that names a setting itself is
    refused."""

    matrix = substitute(json.loads(Path(path).read_text()), values or {})
    cases = matrix.get("cases", [])
    if not isinstance(matrix.get("defaults", {}), dict) or not all(
        isinstance(case, dict) for case in cases
    ):
        raise ValueError(f"{path}: defaults and cases are mappings")
    grids = matrix.get("grid", {})
    named = set(matrix.get("defaults", {}))
    for axes in [*(grids if isinstance(grids, list) else [grids]), *cases]:
        named |= set(axes)
    shared = sorted(named & set(settings))
    if shared:
        raise ValueError(
            f"{path} names {', '.join(shared)}, which the experiment's settings set"
        )
    return {**matrix, "defaults": {**settings, **matrix.get("defaults", {})}}


def load_spec(
    path: Path, output: Path | None = None, variables: dict[str, str] | None = None
) -> dict:
    """The spec, checked: every variable has a value, from ``variables`` or
    the experiment's record, and every sweep names an engine and a matrix
    that leaves the experiment's settings to it, has a name of its own, and
    an interpreter resolved from its own entry, the spec's for its engine,
    or the environment the engine names."""

    spec = json.loads(Path(path).read_text())
    unknown = set(spec) - SPEC_KEYS
    if unknown or spec.get("schema_version") != 1:
        raise ValueError("expected schema_version=1 and gpus, python and sweeps")
    if output is not None:
        spec["output"] = str(Path(output).resolve())
    else:
        folder = Path(path).resolve().parent
        spec["output"] = str((folder / spec.get("output", "output")).resolve())
    if not isinstance(spec.get("gpus"), int) or spec["gpus"] < 1:
        raise ValueError("gpus must be a positive integer")
    pythons = spec.get("python", {})
    if not isinstance(pythons, dict):
        raise ValueError("python maps an engine to its interpreter")
    settings = spec.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings maps a case's field to its value")
    declared = spec.get("variables", {})
    if not isinstance(declared, dict) or not all(
        isinstance(name, str)
        and VARIABLE_NAME.fullmatch(name)
        and isinstance(meaning, str)
        and meaning
        for name, meaning in declared.items()
    ):
        raise ValueError("variables maps each variable's name to what its value is")
    values = {**recorded_values(Path(spec["output"])), **(variables or {})}
    unknown = sorted(set(values) - set(declared))
    if variables and set(variables) - set(declared):
        raise ValueError(f"the spec declares no variable {unknown}")
    missing = sorted(set(declared) - set(values))
    if missing:
        raise ValueError(
            "give every variable the spec declares a value: "
            + "; ".join(f"--variable {name}=<{declared[name]}>" for name in missing)
        )
    values = {name: values[name] for name in declared}
    spec["variables"] = {
        name: {"description": declared[name], "value": values[name]}
        for name in declared
    }
    settings = spec["settings"] = substitute(settings, values)
    budgets = spec.get("summary_budgets_gib", [])
    if not isinstance(budgets, list) or any(
        isinstance(value, bool) or not isinstance(value, int | float) or value <= 0
        for value in budgets
    ):
        raise ValueError("summary_budgets_gib lists GPU-memory budgets in GiB")
    timeout = spec.get("case_timeout", 3600)
    if not isinstance(timeout, int | float) or timeout <= 0:
        raise ValueError("case_timeout must be positive")
    sweeps = spec.get("sweeps")
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("at least one sweep is required")
    names = set()
    for sweep in sweeps:
        unknown = set(sweep) - SWEEP_KEYS
        if unknown or not sweep.get("engine") or not sweep.get("config"):
            raise ValueError("each sweep names an engine and a config")
        sweep.setdefault("name", f"{sweep['engine']}-{Path(sweep['config']).stem}")
        if sweep["name"] in names or "/" in sweep["name"]:
            raise ValueError(f"sweep names are distinct plain names: {sweep['name']}")
        names.add(sweep["name"])
        python = sweep.get("python", pythons.get(sweep["engine"]))
        if not python:
            python = engine_module(sweep["engine"]).ENVIRONMENT
        sweep["python"] = str(interpreter(python))
        sweep.setdefault("case_timeout", timeout)
        sweep["config"] = resolve_config(sweep["config"], Path(path).resolve().parent)
        sweep_matrix(matrix_source(sweep), settings, values)
    return spec


def sweep_command(
    spec: dict,
    sweep: dict,
    matrix: Path,
    output: Path,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
):
    """The sweep launcher's command line for one sweep over its written
    matrix, resuming when its manifest exists."""

    command = [
        sys.executable,
        "-m",
        "baselines.sweep",
        "--engine",
        sweep["engine"],
        "--config",
        str(matrix),
        "--output",
        str(output),
        "--gpus",
        str(spec["gpus"]),
        "--python",
        sweep["python"],
        "--continue-on-error",
        "--case-timeout",
        str(sweep["case_timeout"]),
    ]
    if (output / "manifest.json").exists():
        command.append("--resume")
        if retry_failed:
            command.append("--retry-failed")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def report_command(outputs: list[Path], report: Path):
    return [
        sys.executable,
        "-m",
        "baselines.report",
        *(str(path) for path in outputs),
        "--output",
        str(report),
        "--memory",
        "reserved",
    ]


def run_process(command, log: Path, dry_run: bool) -> int | None:
    """Run one launcher with its console in ``log``; nothing in a dry run."""

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        handle.flush()
        if dry_run:
            return None
        return subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        ).returncode


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


class Console:
    """The experiment's console: each line printed and kept in the run's
    log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a")

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run_sweep(command, log: Path, console: Console) -> int:
    """Run one sweep launcher to its end; every line of its console goes to
    ``log`` and, indented, to the experiment's."""

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                console("  " + line.rstrip("\n"))
        finally:
            returncode = process.wait()
    return returncode


def run_stamp(output: Path) -> str:
    """A new run's name: its local start time, unique among the runs."""

    base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp, number = base, 1
    while any(
        (output / "runs" / f"{stamp}{suffix}").exists() for suffix in (".json", ".log")
    ):
        number += 1
        stamp = f"{base}_{number}"
    return stamp


def merged_record(output: Path, spec: dict, run: dict) -> dict:
    """``experiment.json``: the latest spec, each sweep's latest run, a
    sweep never run marked so, and every run of the experiment."""

    path = output / "experiment.json"
    previous = json.loads(path.read_text()) if path.is_file() else {}
    held = {sweep["name"]: sweep for sweep in previous.get("sweeps", [])}
    for sweep in spec["sweeps"]:
        held.setdefault(
            sweep["name"],
            {
                **sweep,
                "output": str(output / "sweeps" / sweep["name"]),
                "status": "not run",
            },
        )
    for entry in run["sweeps"]:
        held[entry["name"]] = {**entry, "run": run["run"]}
    names = [sweep["name"] for sweep in spec["sweeps"]]
    runs = [item for item in previous.get("runs", []) if item.get("run") != run["run"]]
    runs.append(
        {
            "run": run["run"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "dry_run": run["dry_run"],
            "selected": run["selected"],
            "record": f"runs/{run['run']}.json",
            "console": f"runs/{run['run']}.log",
        }
    )
    return {
        "schema_version": 1,
        "spec_path": run["spec_path"],
        "spec": spec,
        "sweeps": [held[name] for name in names]
        + [entry for name, entry in held.items() if name not in names],
        "runs": runs,
        "report": run["report"]
        if run["report"] is not None
        else previous.get("report"),
        "summary": (
            run["summary"] if run["summary"] is not None else previous.get("summary")
        ),
    }


def case_count(spec: dict, sweep: dict) -> str:
    """``25 cases``, from the sweep's matrix as the sweep expands it."""

    try:
        matrix = sweep_matrix(
            matrix_source(sweep), spec["settings"], variable_values(spec)
        )
        settings = engine_module(sweep["engine"]).Settings
        return f"{len(sweep_runner.expand(matrix, spec['gpus'], settings))} cases"
    except (ValueError, ImportError) as error:
        return f"a matrix the sweep refuses ({error})"


def host_gib() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def header(console: Console, spec: dict, spec_path: Path, selected: list, output: Path):
    """What the run is of: the spec, its settings, every sweep with its case
    count and whether this run runs it."""

    console(f"EXPERIMENT {spec_path}")
    console(f"  output: {output}")
    console(f"  start: {sweep_runner.clock()}; this run's console: {console.path}")
    memory = host_gib()
    console(
        f"  GPUs: {spec['gpus']}"
        + ("" if memory is None else f"; host memory {memory:.0f} GiB")
    )
    shared = " ".join(
        f"{key}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in spec["settings"].items()
    )
    for line in textwrap.wrap(
        shared,
        width=96,
        initial_indent="  settings: ",
        subsequent_indent=" " * 12,
        break_long_words=False,
        break_on_hyphens=False,
    ):
        console(line)
    for sweep in spec["sweeps"]:
        role = "runs" if sweep in selected else "kept as it is"
        console(
            f"  {sweep['name']}: {case_count(spec, sweep)} of {sweep['engine']},"
            f" trainers in {sweep['python']}; {role}"
        )


def spec_path_of(path: Path) -> str:
    """The spec's path, as the repository names it when it is inside."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


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
        console("  dry run: each sweep validates its cases and writes their commands")
    if args.overwrite:
        for sweep in selected:
            held = output / "sweeps" / sweep["name"]
            if held.exists():
                if args.dry_run:
                    console(f"  overwrite: would discard {held}")
                else:
                    shutil.rmtree(held)
                    console(f"  overwrite: discarded {held}")
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
            console(f"ENGINE [{index}/{len(selected)}] {sweep['name']}")
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
                f"ENGINE {sweep['name']}: {status}"
                + (f" (exit status {returncode})" if returncode else "")
                + f"; {sweep_runner.duration(time.monotonic() - sweep_started)},"
                f" {clock_began} to {sweep_runner.clock()}"
            )
            write_json(record_path, run)
            write_json(output / "experiment.json", merged_record(output, spec, run))
        console()
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
                console(f"REPORT {status}: {output / 'report' / 'index.html'}")
            else:
                console("REPORT not drawn: no rows yet")
            console()
        try:
            summary = summaries.write(output, summaries.summarize(output))
            if args.publish is not None:
                summaries.publish(output, summary, args.publish)
        except (ValueError, OSError, KeyError) as error:
            console(f"SUMMARY failed: {error}")
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
                console(f"PUBLISHED {args.publish / 'summary.md'}")
        ending = "finished" + (f", failed: {', '.join(failed)}" if failed else "")
    finally:
        run["finished_at"] = now()
        write_json(record_path, run)
        write_json(output / "experiment.json", merged_record(output, spec, run))
        console()
        console(
            f"EXPERIMENT {ending} at {sweep_runner.clock()} after"
            f" {sweep_runner.duration(time.monotonic() - started)}"
        )
        console(f"  record: {output / 'experiment.json'}; this run: {record_path}")
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


if __name__ == "__main__":
    main()
