"""Run a matrix of independent, resumable cases of one harness on one node.

    python -m baselines.sweep --harness fsdp2 --config matrix.json \\
        --output DIR --gpus N [--python PATH] [--dry-run] [--resume] \\
        [--retry-failed] [--continue-on-error] [--case-timeout SECONDS] \\
        [--limit N]

A matrix names ``defaults`` and either a Cartesian ``grid``, or a list of
grids whose cases are their union, or explicit ``cases`` over the harness's
settings. Every case is validated, refused if
invalid, and named by a digest of what runs, so two spellings of one
configuration are one case. Each attempt of a case is its own process under
``cases/<name>/attempt-NNN/`` with its command, its console log and, when it
completed, the ``row.json`` the harness wrote; a case that ran out of device
memory, failed or timed out gets a row from this launcher saying so.
Out-of-memory is a boundary of the frontier and the sweep continues past it;
any other failure stops the sweep for diagnosis unless told to continue.
``manifest.json`` records the harness, the sources, the environment and
every case, and a resume must match it exactly, so the rows under one output
are comparable.

A harness is the module ``baselines.<harness>.settings``: its ``Settings``
(``from_dict``, ``as_dict``, ``validate``, ``arguments`` and
``describe``), its ``command`` and the ``SOURCE_PATHS`` whose digests
identify its code.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import itertools
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

from baselines.common import write_json
from baselines.common.provenance import ROOT, environment_identity, source_identity
from baselines.common.results import row as result_row
from baselines.common.results import validate_row, varied

INDEX_COLUMNS = (
    "Status",
    "s/update",
    "tok/s",
    "TFLOP/s/GPU",
    "GPU peak GiB",
    "Host peak GiB",
    "Evidence",
)


def harness_module(name):
    """The settings module of a harness, checked for what the sweep calls."""

    try:
        module = import_module(f"baselines.{name}.settings")
    except ModuleNotFoundError as error:
        raise ValueError(f"no harness {name!r}: {error}") from error
    missing = [
        attribute
        for attribute in ("Settings", "command", "SOURCE_PATHS")
        if not hasattr(module, attribute)
    ]
    if missing:
        raise ValueError(f"harness {name!r} lacks {', '.join(missing)}")
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def expand(specification, world_size, settings_type):
    """``[(name, settings)]`` for a matrix: validated, refused if invalid,
    and deduplicated by what runs."""

    unknown = set(specification) - {"schema_version", "defaults", "grid", "cases"}
    if unknown or specification.get("schema_version") != 1:
        raise ValueError("expected schema_version=1 and defaults/grid or cases")
    defaults = specification.get("defaults", {})
    settings_type.from_dict(defaults)
    if "grid" in specification and "cases" in specification:
        raise ValueError("choose a Cartesian grid or an explicit cases list")
    if "cases" in specification:
        overrides = specification["cases"]
    else:
        grids = specification.get("grid", {})
        grids = grids if isinstance(grids, list) else [grids]
        if not grids or any(
            not isinstance(grid, dict)
            or any(
                not isinstance(values, list) or not values for values in grid.values()
            )
            for grid in grids
        ):
            raise ValueError("each grid axis must be a nonempty list")
        overrides = [
            dict(zip(grid, values, strict=True))
            for grid in grids
            for values in itertools.product(*grid.values())
        ]
    if not isinstance(overrides, list) or not overrides:
        raise ValueError("at least one case is required")
    result, seen = [], set()
    for override in overrides:
        settings = settings_type.from_dict({**defaults, **override})
        settings.validate(world_size)
        identity = digest(settings.as_dict())[:12]
        if identity in seen:
            continue
        seen.add(identity)
        result.append((f"case-{len(result) + 1:04d}-{identity}", settings))
    return result


def classify(attempt, returncode, world_size, settings):
    """What an attempt produced: an out-of-memory, a failure, or the
    measurements of a row the harness wrote for exactly this case."""

    failures = [
        json.loads(path.read_text())
        for path in sorted(attempt.glob("rank-*/failure.json"))
    ]
    if any(item.get("status") == "cuda_out_of_memory" for item in failures):
        return {"status": "cuda_out_of_memory", "failures": failures}
    if returncode:
        return {
            "status": "failed",
            "failures": failures,
            "reason": f"exit status {returncode}",
        }
    path = attempt / "row.json"
    if not path.is_file():
        return {"status": "failed", "reason": "the process exited without a row"}
    try:
        row = json.loads(path.read_text())
        validate_row(row)
    except ValueError as error:
        return {"status": "failed", "reason": f"invalid row: {error}"}
    if row["status"] != "completed":
        return {"status": "failed", "reason": f"the row says {row['status']}"}
    expected = settings.describe(world_size)
    for key in ("model", "geometry", "precision", "configuration"):
        if row[key] != expected[key]:
            return {
                "status": "failed",
                "reason": f"the executed {key} differs from the request",
            }
    measurement, memory = row["measurement"], row["memory"]
    return {
        "status": "completed",
        "step_seconds": measurement["step_seconds"],
        "global_tokens_per_second": measurement["global_tokens_per_second"],
        "useful_tflops_per_gpu": measurement.get("useful_tflops_per_gpu"),
        "performance_result": bool(measurement.get("performance_result", True)),
        "peak_allocated_bytes": memory["peak_allocated_bytes"],
        "peak_reserved_bytes": memory["peak_reserved_bytes"],
        "host_peak_rss_bytes": memory["host_peak_rss_bytes"],
    }


def latest(case):
    attempts = sorted(case.glob("attempt-*"))
    if not attempts:
        return {"status": "pending"}
    path = attempts[-1] / "outcome.json"
    if not path.exists():
        return {"status": "interrupted", "attempt": attempts[-1].name}
    return json.loads(path.read_text())


def failure_row(attempt, outcome, settings, world_size, harness, manifest):
    """The row this launcher writes for an attempt that produced none: the
    request as described by the settings, the hardware the first rank
    reported before it failed, and what went wrong."""

    hardware = {}
    request = attempt / "rank-00000" / "request.json"
    if request.is_file():
        hardware = json.loads(request.read_text()).get("hardware") or {}
    failure = None
    if outcome["status"] in ("failed", "timeout", "cuda_out_of_memory"):
        failure = {
            key: outcome.get(key) for key in ("returncode", "reason", "failures")
        }
    return result_row(
        harness=harness,
        status=outcome["status"],
        **settings.describe(world_size),
        hardware=hardware,
        provenance={
            "source": manifest["source"],
            "environment": manifest["environment"],
        },
        failure=failure,
    )


def _cell(value, form):
    return "—" if value is None else form.format(value)


def report(output, cases, harness, world_size):
    """``results.json`` and the ``README.md`` index of every case's latest
    outcome, with a column per setting the matrix varies."""

    rows, described = [], {}
    for name, settings in cases:
        description = settings.describe(world_size)
        described[name] = {**description["geometry"], **description["configuration"]}
    knobs = varied(described)
    text = [
        f"# {harness} sweep",
        "",
        "[Exact manifest and commands](manifest.json). Throughput is global; "
        "device peaks are PyTorch allocator measurements; a completed run that "
        "was a qualification or a profile is not a throughput result.",
        "",
        "| Case | " + " | ".join([*knobs, *INDEX_COLUMNS]) + " |",
        "|---|" + "---|" * (len(knobs) + len(INDEX_COLUMNS)),
    ]
    for name, settings in cases:
        outcome = latest(output / "cases" / name)
        rows.append({"case": name, "settings": settings.as_dict(), **outcome})
        status = outcome["status"]
        eligible = status == "completed" and outcome.get("performance_result", False)
        if status == "completed" and not eligible:
            status += " (qualification/profile)"
        evidence = f"[config](cases/{name}/config.json)"
        if "attempt" in outcome:
            path = f"cases/{name}/{outcome['attempt']}"
            evidence += f" · [log]({path}/console.log) · [outcome]({path}/outcome.json)"
            if (output / path / "row.json").is_file():
                evidence += f" · [row]({path}/row.json)"
        cells = [
            name,
            *(json.dumps(described[name].get(key)) for key in knobs),
            status,
            _cell(outcome.get("step_seconds") if eligible else None, "{:.4f}"),
            _cell(
                outcome.get("global_tokens_per_second") if eligible else None, "{:,.0f}"
            ),
            _cell(outcome.get("useful_tflops_per_gpu") if eligible else None, "{:.1f}"),
            _cell(outcome.get("peak_allocated_bytes"), "{:.2f}")
            if outcome.get("peak_allocated_bytes") is None
            else f"{outcome['peak_allocated_bytes'] / 2**30:.2f}",
            _cell(outcome.get("host_peak_rss_bytes"), "{:.2f}")
            if outcome.get("host_peak_rss_bytes") is None
            else f"{outcome['host_peak_rss_bytes'] / 2**30:.2f}",
            evidence,
        ]
        text.append("| " + " | ".join(cells) + " |")
    write_json(
        output / "results.json",
        {"harness": harness, "world_size": world_size, "cases": rows},
    )
    (output / "README.md").write_text("\n".join(text) + "\n")


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_case(command_line, attempt, settings, world_size, timeout):
    """Run one attempt to completion, timeout or interruption, recording its
    console, its process and its outcome beside it."""

    started = time.monotonic()
    environment = os.environ.copy()
    threads = getattr(settings, "cpu_threads", None)
    if threads is not None:
        environment["OMP_NUM_THREADS"] = str(threads)
        environment["MKL_NUM_THREADS"] = str(threads)
    with (attempt / "console.log").open("w") as log:
        process = subprocess.Popen(
            command_line,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            attempt / "process.json", {"pid": process.pid, "command": command_line}
        )
        try:
            returncode = process.wait(timeout=timeout)
            try:
                outcome = classify(attempt, returncode, world_size, settings)
            except (ValueError, KeyError, TypeError) as error:
                outcome = {"status": "failed", "reason": f"invalid result: {error}"}
            outcome["returncode"] = returncode
        except subprocess.TimeoutExpired:
            stop(process)
            outcome = {"status": "timeout", "reason": f"exceeded {timeout} seconds"}
        except BaseException:
            stop(process)
            write_json(
                attempt / "outcome.json",
                {"status": "interrupted", "attempt": attempt.name},
            )
            raise
    outcome.update(
        {
            "attempt": attempt.name,
            "wall_seconds": time.monotonic() - started,
            "finished_at": datetime.datetime.now().astimezone().isoformat(),
        }
    )
    write_json(attempt / "outcome.json", outcome)
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--harness", required=True, help="baselines.<harness>")
    parser.add_argument("--config", required=True, type=Path, help="the matrix")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gpus", type=int, required=True, help="single-node torchrun world size"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
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
        module = harness_module(args.harness)
        cases = expand(json.loads(args.config.read_text()), args.gpus, module.Settings)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another sweep launcher holds this output directory")
        manifest = {
            "schema_version": 1,
            "harness": args.harness,
            "world_size": args.gpus,
            "python": str(args.python.resolve()),
            "source": source_identity(["baselines/sweep.py", *module.SOURCE_PATHS]),
            "environment": environment_identity(args.python),
            "cases": [
                {"name": name, "settings": settings.as_dict()}
                for name, settings in cases
            ],
        }
        path = output / "manifest.json"
        if path.exists():
            if not args.resume or json.loads(path.read_text()) != manifest:
                parser.error(
                    "existing output requires --resume with the same harness, "
                    "cases, world size, sources and Python"
                )
        else:
            write_json(path, manifest)
        for name, settings in cases:
            case = output / "cases" / name
            write_json(case / "config.json", settings.as_dict())
            first = module.command(
                args.python, args.gpus, settings, case / "attempt-001"
            )
            (case / "command.txt").write_text(shlex.join(first) + "\n")
        report(output, cases, args.harness, args.gpus)
        print(f"{len(cases)} configurations; index: {output / 'README.md'}", flush=True)
        if args.dry_run:
            return
        attempted = 0
        for name, settings in cases:
            case = output / "cases" / name
            previous = latest(case)
            if previous["status"] == "completed" or (
                previous["status"] != "pending" and not args.retry_failed
            ):
                continue
            attempts = list(case.glob("attempt-*"))
            number = (
                max((int(path.name.split("-")[-1]) for path in attempts), default=0) + 1
            )
            attempt = case / f"attempt-{number:03d}"
            attempt.mkdir()
            cmd = module.command(args.python, args.gpus, settings, attempt)
            (attempt / "command.txt").write_text(shlex.join(cmd) + "\n")
            print(
                f"Starting {name}/{attempt.name}: {shlex.join(settings.arguments())}",
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
                            args.harness,
                            manifest,
                        ),
                    )
            finally:
                report(output, cases, args.harness, args.gpus)
            print(f"{name}: {outcome['status']}", flush=True)
            attempted += 1
            if (
                outcome["status"] not in ("completed", "cuda_out_of_memory")
                and not args.continue_on_error
            ):
                raise SystemExit(f"Stopped for diagnosis: {attempt / 'console.log'}")
            if args.limit is not None and attempted >= args.limit:
                break


if __name__ == "__main__":
    main()
