"""Run independent, resumable FSDP2 configurations on one torchrun node."""

from __future__ import annotations

import argparse
import dataclasses
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
from pathlib import Path

from baselines.common import summarize_cycles, write_json
from baselines.fsdp2.settings import Settings

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def expand(specification, world_size):
    unknown = set(specification) - {"schema_version", "defaults", "grid", "cases"}
    if unknown or specification.get("schema_version") != 1:
        raise ValueError("expected schema_version=1 and defaults/grid or cases")
    defaults = specification.get("defaults", {})
    Settings.from_dict(defaults)
    if "grid" in specification and "cases" in specification:
        raise ValueError("choose a Cartesian grid or an explicit cases list")
    if "cases" in specification:
        overrides = specification["cases"]
    else:
        grid = specification.get("grid", {})
        if not isinstance(grid, dict) or any(
            not isinstance(values, list) or not values for values in grid.values()
        ):
            raise ValueError("each grid axis must be a nonempty list")
        overrides = [
            dict(zip(grid, values, strict=True))
            for values in itertools.product(*grid.values())
        ]
    if not isinstance(overrides, list) or not overrides:
        raise ValueError("at least one case is required")
    result, seen = [], set()
    for override in overrides:
        settings = Settings.from_dict({**defaults, **override})
        accumulation = settings.validate(world_size)
        # Derived and explicitly specified equivalent accumulation share an ID.
        settings = dataclasses.replace(settings, gradient_accumulation=accumulation)
        identity = digest(dataclasses.asdict(settings))[:12]
        if identity in seen:
            continue
        seen.add(identity)
        result.append((f"case-{len(result) + 1:04d}-{identity}", settings))
    return result


def source_identity():
    """Do not reuse old timings after local trainer/model source has changed."""
    paths = list((ROOT / "baselines/fsdp2").rglob("*.py"))
    paths.extend((ROOT / "baselines/common").rglob("*.py"))
    paths.append(ROOT / "baselines/__init__.py")
    paths.extend((ROOT / "workloads/models").rglob("*.py"))
    files = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    return {"git_head": revision.stdout.strip(), "file_sha256": files}


def environment_identity(python):
    """Probe the chosen interpreter without importing Torch or initializing CUDA."""
    probe = """
import hashlib, importlib.metadata, importlib.util, json, pathlib, sys
packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
spec = importlib.util.find_spec('mlops')
files = {}
if spec is not None and spec.origin:
    root = pathlib.Path(spec.origin).parent
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*.py'))}
print(json.dumps({'python': sys.version, 'packages': packages, 'mlops_source': files}))
"""
    result = subprocess.run(
        [str(python), "-c", probe], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def command(python, world_size, settings, output):
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        "-m",
        "baselines.fsdp2.train",
        "--output",
        str(output),
        *settings.arguments(),
    ]


def classify(attempt, returncode, world_size, settings):
    failures = [
        json.loads(path.read_text())
        for path in sorted(attempt.glob("rank-*/failure.json"))
    ]
    if any(row.get("status") == "cuda_out_of_memory" for row in failures):
        return {"status": "cuda_out_of_memory", "failures": failures}
    if returncode:
        return {"status": "failed", "failures": failures}
    result_path = attempt / "results.json"
    if not result_path.is_file():
        return {"status": "failed", "reason": "process exited without all-rank results"}
    result = json.loads(result_path.read_text())
    ranks = result.get("ranks", [])
    if (
        result.get("status") != "completed"
        or len(ranks) != world_size
        or {row.get("rank") for row in ranks} != set(range(world_size))
        or any(
            row.get("status") != "completed"
            or len(row.get("cycles", [])) != settings.steps
            for row in ranks
        )
    ):
        return {"status": "failed", "reason": "incomplete rank measurements"}
    request = result.get("request", {})
    if any(
        request.get(key) != value for key, value in dataclasses.asdict(settings).items()
    ):
        return {
            "status": "failed",
            "reason": "executed configuration differs from request",
        }
    summary = summarize_cycles(
        [row["cycles"] for row in ranks],
        global_tokens=settings.global_tokens,
        useful_flops_per_step=result.get("useful_flops_per_step"),
    )
    return {
        "status": "completed",
        **summary,
        "performance_result": all(row["performance_result"] for row in ranks),
        "qualified_against_reference": all(
            row["qualified_against_reference"] for row in ranks
        ),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in ranks),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in ranks),
        "host_peak_rss_bytes_sum": sum(row["host_peak_rss_bytes"] for row in ranks),
    }


def latest(case):
    attempts = sorted(case.glob("attempt-*"))
    if not attempts:
        return {"status": "pending"}
    path = attempts[-1] / "outcome.json"
    if not path.exists():
        return {"status": "interrupted", "attempt": attempts[-1].name}
    return json.loads(path.read_text())


def report(output, cases):
    rows = []
    text = [
        "# FSDP2 sweep",
        "",
        "[Exact manifest and commands](manifest.json). Microbatch tokens are per rank; "
        "throughput is global. GPU peaks are PyTorch allocator measurements. "
        "Qualification/profiled runs are not throughput results.",
        "",
        "| Case | MB tokens | Accum | Recompute | Offload | Prefetch F/B | Reshard F/B "
        "| Sync | Cap GiB | Status | s/update | tok/s | GPU peak GiB | Evidence |",
        "|---|---:|---:|---:|---|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for name, settings in cases:
        outcome = latest(output / "cases" / name)
        row = {"case": name, "settings": dataclasses.asdict(settings), **outcome}
        rows.append(row)
        eligible = outcome.get("performance_result", False)
        seconds = f"{outcome['step_seconds']:.4f}" if eligible else "—"
        throughput = f"{outcome['global_tokens_per_second']:,.0f}" if eligible else "—"
        peak = (
            f"{outcome['peak_allocated_bytes'] / 2**30:.2f}"
            if "peak_allocated_bytes" in outcome
            else "—"
        )
        status = outcome["status"]
        if status == "completed" and not eligible:
            status += " (qualification/profile)"
        evidence = f"[config](cases/{name}/config.json)"
        if "attempt" in outcome:
            path = f"cases/{name}/{outcome['attempt']}"
            evidence += f" · [log]({path}/console.log) · [outcome]({path}/outcome.json)"
        text.append(
            f"| {name} | {settings.microbatch_tokens} "
            f"| {settings.gradient_accumulation} "
            f"| {settings.recompute:.0%} | {settings.offload} "
            f"| {settings.forward_prefetch}/{settings.backward_prefetch} "
            f"| {settings.reshard_after_forward}/{settings.reshard_after_backward} "
            f"| {settings.gradient_sync} | {settings.allocator_cap_gib or 'none'} "
            f"| {status} | {seconds} | {throughput} | {peak} | {evidence} |"
        )
    write_json(output / "results.json", {"cases": rows})
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
    started = time.monotonic()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(settings.cpu_threads)
    environment["MKL_NUM_THREADS"] = str(settings.cpu_threads)
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
            outcome = {"status": "timeout"}
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gpus", type=int, default=8, help="Single-node torchrun world size"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs/commands/index without launching GPUs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same manifest without overwriting attempts",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="New attempts for failed/OOM/timeout cases",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after non-OOM errors; default stops for diagnosis",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=3600,
        help="Seconds including setup and compilation",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum new attempts in this invocation"
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
    cases = expand(json.loads(args.config.read_text()), args.gpus)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another sweep launcher holds this output directory")
        manifest = {
            "schema_version": 1,
            "world_size": args.gpus,
            "python": str(args.python.resolve()),
            "source": source_identity(),
            "environment": environment_identity(args.python),
            "cases": [
                {"name": name, "settings": dataclasses.asdict(settings)}
                for name, settings in cases
            ],
        }
        path = output / "manifest.json"
        if path.exists():
            if not args.resume or json.loads(path.read_text()) != manifest:
                parser.error(
                    "existing output requires --resume with identical "
                    "settings, source, and Python"
                )
        else:
            write_json(path, manifest)
        for name, settings in cases:
            case = output / "cases" / name
            write_json(case / "config.json", dataclasses.asdict(settings))
            (case / "command.txt").write_text(
                shlex.join(
                    command(args.python, args.gpus, settings, case / "attempt-001")
                )
                + "\n"
            )
        report(output, cases)
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
            cmd = command(args.python, args.gpus, settings, attempt)
            (attempt / "command.txt").write_text(shlex.join(cmd) + "\n")
            print(
                f"Starting {name}/{attempt.name}: {shlex.join(settings.arguments())}",
                flush=True,
            )
            try:
                outcome = run_case(cmd, attempt, settings, args.gpus, args.case_timeout)
            finally:
                report(output, cases)
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
