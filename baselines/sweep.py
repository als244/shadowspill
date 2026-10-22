"""Run a matrix of independent, resumable cases of one engine on one node.

    python -m baselines.sweep --engine fsdp2 --config matrix.json \\
        --output DIR --gpus N [--python PATH] [--dry-run] [--resume] \\
        [--retry-failed] [--continue-on-error] [--case-timeout SECONDS] \\
        [--limit N]

A matrix names ``defaults`` and either a Cartesian ``grid``, or a list of
grids whose cases are their union, or explicit ``cases`` over the engine's
settings. Every case is validated, refused if invalid, and identified by a
digest of what runs, so two spellings of one configuration are one case. A
case's folder is named for the settings its matrix varies, each as a short
code and its value (``CODES``; a setting without a code goes by its name),
then eight characters of that digest: ``mb16384_rc0.25_act-cpu__1a2b3c4d``.
Each attempt of a case is its own process under ``cases/<name>/attempt-NNN/``
with its command, its console log and, when it completed, the ``row.json``
the engine wrote; a case that ran out of device
memory, failed or timed out gets a row from this launcher saying so.
Out-of-memory is a boundary of the frontier and the sweep continues past it;
any other failure stops the sweep for diagnosis unless told to continue.
``manifest.json`` records the engine, the sources, the environment and
every case. A resume must find the same engine, world size and
interpreter, with the same Python and packages; which code ran is the
caller's to track, and each row records its sources. The matrix itself may
have changed. A case keeps
the name its settings were first given there, by the digest that ends it,
and a new case is numbered after every case the output holds, so a resume
over a grown matrix attempts only the cases it adds, and the manifest names
the matrix's cases, the ones its rows count for. A resume attempts every
named case with no outcome, never attempted or interrupted, keeps the
completed ones, and attempts the failed, out-of-memory and timed-out ones
again only with ``--retry-failed``.

The console names the cases, the settings the matrix varies and the
outcomes so far, then gives a line as each case starts, with its settings,
and a line as it ends, with its outcome and, when it completed, its
throughput, update time, GPU peak held and in use, and host peak; the last
line counts every case's outcome and the run's duration.

An engine is the module ``baselines.<engine>.settings``: its ``Settings``
(``from_dict``, ``as_dict``, ``validate``, ``arguments``, ``describe`` and
``resolve``), the ``SOURCE_PATHS`` whose digests identify its code and the
``ENVIRONMENT`` its trainer runs in. Every attempt is launched through
``baselines.common.launch``, which resolves the case in this interpreter and
starts the engine's trainer in ``--python``, by default the environment the
engine names.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

from baselines.common import launch, write_json
from baselines.common.provenance import ROOT, environment_identity, source_identity
from baselines.common.results import row as result_row
from baselines.common.results import validate_row, varied
from baselines.report import DERIVED_GEOMETRY

#: The short code a case's folder name gives a setting; any other setting
#: goes by its name.
CODES = {
    "tokens_per_microbatch": "mb",
    "recompute": "rc",
    "activation_offload": "act",
    "offload_policy": "op",
    "pin_memory": "pin",
    "forward_prefetch": "fpf",
    "backward_prefetch": "bpf",
    "reshard_after_forward": "raf",
    "reshard_after_backward": "rab",
    "gradient_sync": "gs",
    "compile_blocks": "cb",
    "float8": "f8",
    "optimizer": "opt",
    "offload_gradients": "og",
    "zero_stage": "z",
    "offload_param": "ofp",
    "offload_optimizer": "ofo",
    "deepcompile": "dc",
    "execution_gib": "ex",
    "spill_gib": "sp",
    "resolution_options": "res",
}

#: How an outcome reads on the console, in the order counts list.
OUTCOME_WORDS = {
    "completed": "completed",
    "cuda_out_of_memory": "out of device memory",
    "failed": "failed",
    "timeout": "timed out",
    "interrupted": "interrupted",
    "pending": "not yet run",
}

INDEX_COLUMNS = (
    "Status",
    "s/update",
    "tok/s",
    "TFLOP/s/GPU",
    "GPU peak GiB",
    "Host peak GiB",
    "Evidence",
)


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
    unique, seen = [], set()
    for override in overrides:
        settings = settings_type.from_dict({**defaults, **override})
        settings.validate(world_size)
        identity = digest(settings.as_dict())
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((identity, settings))
    values = {
        identity: flat(settings.describe(world_size)) for identity, settings in unique
    }
    order = list(CODES)
    knobs = sorted(
        (key for key in varied(values) if key not in DERIVED_GEOMETRY),
        key=lambda key: (order.index(key) if key in order else len(order), key),
    )
    names = [case_name(values[identity], knobs, identity[:8]) for identity, _ in unique]
    if len(set(names)) != len(names):
        raise ValueError("two cases share a name; their digests begin alike")
    return list(zip(names, (settings for _, settings in unique), strict=True))


def flat(description):
    """A case's geometry and configuration as one mapping."""

    return {**description["geometry"], **description["configuration"]}


def case_name(values, knobs, identity):
    """A case's folder name: each setting its matrix varies, as its code and
    value, then the digest of everything it runs."""

    parts = []
    for key in knobs:
        code, value = CODES.get(key, key), values.get(key)
        if isinstance(value, bool):
            parts.append(f"{code}-{'on' if value else 'off'}")
        elif isinstance(value, int):
            parts.append(f"{code}{value}")
        elif isinstance(value, float):
            parts.append(f"{code}{value:g}")
        else:
            text = value if isinstance(value, str) else json.dumps(value)
            parts.append(f"{code}-{re.sub(r'[^A-Za-z0-9.]+', '-', text).strip('-')}")
    return "_".join(parts or ["case"]) + "__" + identity


def keep_names(cases, output):
    """The cases with the names ``output`` already gives their digests: a
    case keeps its folder when a grown matrix varies other settings."""

    held = {
        path.name.rsplit("__", 1)[1]: path.name
        for path in (output / "cases").glob("*__*")
    }
    return [
        (held.get(name.rsplit("__", 1)[1], name), settings) for name, settings in cases
    ]


def comparable(held, manifest):
    """What differs between a manifest an output holds and a resume's that
    makes their rows incomparable: the engine, the world size, the
    interpreter, its version or its packages. The matrix may change, and so
    may the code, ShadowSpill's and mlops's sources included."""

    def environment(value):
        return {
            key: item for key, item in (value or {}).items() if key != "mlops_source"
        }

    differences = [
        key
        for key in ("engine", "world_size", "python")
        if held.get(key) != manifest[key]
    ]
    if environment(held.get("environment")) != environment(manifest["environment"]):
        differences.append("environment")
    return differences


def classify(attempt, returncode, world_size, settings):
    """What an attempt produced: an out-of-memory, a failure, or the
    measurements of a row the engine wrote for exactly this case."""

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


def to_attempt(previous, retry_failed):
    """Whether a case gets a new attempt: one never attempted or interrupted
    does, a completed one never does, and one that failed, ran out of
    memory or timed out does on a retry."""

    status = previous["status"]
    return status in ("pending", "interrupted") or (
        status != "completed" and retry_failed
    )


def latest(case):
    attempts = sorted(case.glob("attempt-*"))
    if not attempts:
        return {"status": "pending"}
    path = attempts[-1] / "outcome.json"
    if not path.exists():
        return {"status": "interrupted", "attempt": attempts[-1].name}
    return json.loads(path.read_text())


def failure_row(attempt, outcome, settings, world_size, engine, manifest):
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
        engine=engine,
        status=outcome["status"],
        **settings.describe(world_size),
        hardware=hardware,
        provenance={
            "source": manifest["source"],
            "environment": manifest["environment"],
        },
        failure=failure,
    )


def described(cases, world_size):
    """Per case, the geometry and configuration its settings describe."""

    return {name: flat(settings.describe(world_size)) for name, settings in cases}


def clock():
    """The local time, as the console shows it."""

    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def duration(seconds):
    """``1h 02m 03s`` or ``2m 03s``."""

    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def tally(statuses):
    """``3 completed, 1 out of device memory, 8 not yet run``."""

    order = list(OUTCOME_WORDS)
    counts = Counter(statuses)
    return ", ".join(
        f"{counts[status]} {OUTCOME_WORDS.get(status, status)}"
        for status in sorted(
            counts, key=lambda item: order.index(item) if item in order else len(order)
        )
    )


def labels(cases, world_size):
    """The settings the matrix varies, those the geometry derives from them
    left out, and per case ``key=value`` for each."""

    values = described(cases, world_size)
    knobs = [key for key in varied(values) if key not in DERIVED_GEOMETRY]
    return knobs, {
        name: " ".join(
            f"{key}={value if isinstance(value, str) else json.dumps(value)}"
            for key in knobs
            for value in [values[name].get(key)]
        )
        for name in values
    }


#: The width of a section's rule, as quickstart draws it.
RULE_WIDTH = 72


def rule(title):
    """A section's heading: ``── title ───``."""

    head = f"── {title} "
    return head + "─" * max(0, RULE_WIDTH - len(head))


def field(name, value):
    """One labeled line of a block, its values aligned."""

    return f"  {name:<13}{value}"


def geometry_text(settings, world_size):
    """A case's microbatches as the console describes them."""

    geometry = settings.geometry(world_size)
    if getattr(settings, "tokens_per_microbatch", 0) is None:
        sizes = settings.microbatch_candidates(world_size)
        return (
            f"searched: {sizes[-1]:,} to {sizes[0]:,} tokens per microbatch,"
            f" {geometry.sequence_length:,}-token sequences"
        )
    return (
        f"{geometry.sequences_per_microbatch} x {geometry.sequence_length:,}-token"
        f" sequences per microbatch, {geometry.accumulation} microbatches per update"
    )


def started_block(position, name, label, settings, world_size, attempt):
    """A case as it starts: what it runs and where its output goes."""

    lines = [
        f"CASE START {position} {name}",
        field("settings", label or "the matrix's defaults"),
        field("geometry", geometry_text(settings, world_size)),
    ]
    if attempt.name != "attempt-001":
        lines.append(field("attempt", f"{attempt.name}, a retry"))
    lines.extend(
        [
            field("log", attempt / "console.log"),
            field("start", clock()),
        ]
    )
    return lines


def ended_block(position, name, outcome, attempt):
    """A case as it ends: its outcome and, when it completed, what it
    measured."""

    status = outcome["status"]
    lines = [f"CASE {OUTCOME_WORDS.get(status, status).upper()} {position} {name}"]
    if status == "completed":
        lines.append(
            field(
                "throughput",
                f"{outcome['global_tokens_per_second']:,.1f} tok/s,"
                f" {outcome['step_seconds']:.2f} s/update",
            )
        )
        lines.append(
            field(
                "memory",
                f"GPU {outcome['peak_reserved_bytes'] / 2**30:.2f} GiB held"
                f" ({outcome['peak_allocated_bytes'] / 2**30:.2f} in use),"
                f" host {outcome['host_peak_rss_bytes'] / 2**30:.2f} GiB",
            )
        )
        if not outcome.get("performance_result", True):
            lines.append(
                field("note", "a qualification or profile, not a throughput result")
            )
    elif outcome.get("reason"):
        lines.append(field("reason", outcome["reason"]))
    if "wall_seconds" in outcome:
        lines.append(
            field("duration", f"{duration(outcome['wall_seconds'])}, ended {clock()}")
        )
    if status != "completed":
        lines.append(field("log", attempt / "console.log"))
    if (attempt / "row.json").is_file():
        lines.append(field("row", attempt / "row.json"))
    return lines


def _cell(value, form):
    return "—" if value is None else form.format(value)


def report(output, cases, engine, world_size):
    """``results.json`` and the ``README.md`` index of every case's latest
    outcome, with a column per setting the matrix varies."""

    rows, values = [], described(cases, world_size)
    knobs = varied(values)
    text = [
        f"# {engine} sweep",
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
            *(json.dumps(values[name].get(key)) for key in knobs),
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
        {"engine": engine, "world_size": world_size, "cases": rows},
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
            "source": source_identity(["baselines/sweep.py", *module.SOURCE_PATHS]),
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
        for name, settings in cases:
            case = output / "cases" / name
            write_json(case / "config.json", settings.as_dict())
            first = launch.command(
                args.engine, args.python, args.gpus, settings, case / "attempt-001"
            )
            (case / "command.txt").write_text(shlex.join(first) + "\n")
        report(output, cases, args.engine, args.gpus)
        knobs, label_of = labels(cases, args.gpus)
        lines = [
            field("cases", f"{len(cases)} at world size {args.gpus}"),
            field("interpreter", args.python),
            field("varying", ", ".join(knobs) if knobs else "nothing"),
        ]
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
    attempted = 0
    for index, (name, settings) in enumerate(cases, 1):
        case = output / "cases" / name
        if not to_attempt(latest(case), args.retry_failed):
            continue
        position = f"[{index:>{width}}/{len(cases)}]"
        attempts = list(case.glob("attempt-*"))
        number = (
            max((int(path.name.split("-")[-1]) for path in attempts), default=0) + 1
        )
        attempt = case / f"attempt-{number:03d}"
        attempt.mkdir()
        cmd = launch.command(args.engine, args.python, args.gpus, settings, attempt)
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


if __name__ == "__main__":
    main()
