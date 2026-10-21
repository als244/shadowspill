"""Run an experiment: the sweeps of several harnesses, in order, on one node,
then the report over everything they wrote.

    python -m baselines.experiment --spec experiment.json [--output DIR] \\
        [--only NAME ...] [--limit N] [--dry-run] [--skip-report]

A spec names the GPU count, the interpreter each harness runs in, the
settings every case of every sweep takes, and the sweeps: each a harness,
its matrix and optionally its own interpreter, timeout and name. The
settings are what the harnesses of one experiment share, the model, the
geometry, the precision contract and the run's length; a sweep's matrix
names only its harness's own settings, and one that names a shared setting
is refused, so no harness runs another model, geometry or contract than the
rest. Each sweep's matrix, with the settings in its defaults, is written to
``<output>/matrices/<name>.json``. Every sweep runs through
``baselines.sweep`` as its own process, with ``--resume`` once its manifest
exists and ``--continue-on-error``, into ``<output>/sweeps/<name>``, so an
experiment interrupted anywhere resumes where it stopped; a sweep's console
goes to ``launcher.log`` beside its cases. ``experiment.json`` records the
spec and every sweep's outcome. The report of ``baselines.report`` is then
drawn over every sweep into ``<output>/report``.

An interpreter is a path, or the name of an environment beside the running
one: ``baselines`` is ``<prefix>/../baselines/bin/python``.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import shlex
import subprocess
import sys
from pathlib import Path

from baselines.common import write_json
from baselines.common.launch import harness_module, interpreter
from baselines.common.provenance import ROOT

SPEC_KEYS = {
    "schema_version",
    "output",
    "gpus",
    "python",
    "case_timeout",
    "settings",
    "sweeps",
}
SWEEP_KEYS = {"name", "harness", "config", "python", "case_timeout"}


def matrix_source(sweep: dict) -> Path:
    """The sweep's matrix file, from the repository root or as given."""

    path = ROOT / sweep["config"]
    return path if path.is_file() else Path(sweep["config"])


def sweep_matrix(path: Path, settings: dict) -> dict:
    """The sweep's matrix with the experiment's settings in its defaults; a
    matrix that names one of them itself is refused."""

    matrix = json.loads(Path(path).read_text())
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


def load_spec(path: Path, output: Path | None = None) -> dict:
    """The spec, checked: every sweep names a harness and a matrix that
    leaves the experiment's settings to it, has a name of its own, and an
    interpreter resolved from its own entry, the spec's for its harness, or
    the environment the harness names."""

    spec = json.loads(Path(path).read_text())
    unknown = set(spec) - SPEC_KEYS
    if unknown or spec.get("schema_version") != 1:
        raise ValueError("expected schema_version=1 and gpus, python and sweeps")
    if output is not None:
        spec["output"] = str(output)
    if not spec.get("output"):
        raise ValueError("the spec or --output names the experiment's directory")
    if not isinstance(spec.get("gpus"), int) or spec["gpus"] < 1:
        raise ValueError("gpus must be a positive integer")
    pythons = spec.get("python", {})
    if not isinstance(pythons, dict):
        raise ValueError("python maps a harness to its interpreter")
    settings = spec.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings maps a case's field to its value")
    timeout = spec.get("case_timeout", 3600)
    if not isinstance(timeout, int | float) or timeout <= 0:
        raise ValueError("case_timeout must be positive")
    sweeps = spec.get("sweeps")
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("at least one sweep is required")
    names = set()
    for sweep in sweeps:
        unknown = set(sweep) - SWEEP_KEYS
        if unknown or not sweep.get("harness") or not sweep.get("config"):
            raise ValueError("each sweep names a harness and a config")
        sweep.setdefault("name", f"{sweep['harness']}-{Path(sweep['config']).stem}")
        if sweep["name"] in names or "/" in sweep["name"]:
            raise ValueError(f"sweep names are distinct plain names: {sweep['name']}")
        names.add(sweep["name"])
        python = sweep.get("python", pythons.get(sweep["harness"]))
        if not python:
            python = harness_module(sweep["harness"]).ENVIRONMENT
        sweep["python"] = str(interpreter(python))
        sweep.setdefault("case_timeout", timeout)
        if not matrix_source(sweep).is_file():
            raise ValueError(f"no matrix at {sweep['config']}")
        sweep_matrix(matrix_source(sweep), settings)
    return spec


def sweep_command(
    spec: dict, sweep: dict, matrix: Path, output: Path, *, limit: int | None = None
):
    """The sweep launcher's command line for one sweep over its written
    matrix, resuming when its manifest exists."""

    command = [
        sys.executable,
        "-m",
        "baselines.sweep",
        "--harness",
        sweep["harness"],
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="overrides the spec's output")
    parser.add_argument(
        "--only", action="append", default=[], metavar="NAME", help="run these sweeps"
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
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    unknown = set(args.only) - {sweep["name"] for sweep in spec["sweeps"]}
    if unknown:
        parser.error(f"no such sweeps: {sorted(unknown)}")
    output = Path(spec["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "experiment.json"
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another experiment holds this output directory")
        record = {
            "schema_version": 1,
            "spec": spec,
            "started_at": datetime.datetime.now().astimezone().isoformat(),
            "dry_run": args.dry_run,
            "sweeps": [],
            "report": None,
        }
        outputs = []
        failed = []
        for sweep in spec["sweeps"]:
            sweep_output = output / "sweeps" / sweep["name"]
            outputs.append(sweep_output)
            if args.only and sweep["name"] not in args.only:
                record["sweeps"].append(
                    {**sweep, "output": str(sweep_output), "status": "skipped"}
                )
                continue
            matrix = output / "matrices" / f"{sweep['name']}.json"
            write_json(matrix, sweep_matrix(matrix_source(sweep), spec["settings"]))
            command = sweep_command(spec, sweep, matrix, sweep_output, limit=args.limit)
            if args.dry_run:
                # The sweep's own dry run expands and validates every case, so
                # a matrix its harness refuses fails here, not mid-experiment.
                command.append("--dry-run")
            started = datetime.datetime.now().astimezone().isoformat()
            print(f"{sweep['name']}: {shlex.join(command)}", flush=True)
            returncode = run_process(command, sweep_output / "launcher.log", False)
            status = (
                "failed"
                if returncode != 0
                else ("planned" if args.dry_run else "completed")
            )
            if status == "failed":
                failed.append(sweep["name"])
            record["sweeps"].append(
                {
                    **sweep,
                    "output": str(sweep_output),
                    "command": command,
                    "status": status,
                    "returncode": returncode,
                    "started_at": started,
                    "finished_at": datetime.datetime.now().astimezone().isoformat(),
                }
            )
            write_json(record_path, record)
            print(f"{sweep['name']}: {status}", flush=True)
        if not args.skip_report:
            drawn = (
                [path for path in outputs if path.is_dir()]
                if not args.dry_run
                else outputs
            )
            command = report_command(drawn, output / "report")
            returncode = run_process(command, output / "report.log", args.dry_run)
            record["report"] = {
                "output": str(output / "report"),
                "command": command,
                "status": "planned"
                if args.dry_run
                else ("completed" if returncode == 0 else "failed"),
                "returncode": returncode,
            }
            if returncode not in (None, 0):
                failed.append("report")
        record["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        write_json(record_path, record)
        print(f"experiment record: {record_path}", flush=True)
        if failed:
            raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
