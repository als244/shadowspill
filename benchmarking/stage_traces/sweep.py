"""Capture unique stages, then drain a queue with one Nsight process per GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def stamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def visible_devices(selection):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selection != ["all"]:
        if visible is None:
            return selection
        available = visible.split(",")
        # Explicit values name ordinals in the current visible list.
        return [available[int(value)] for value in selection]
    if visible:
        return visible.split(",")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    )
    return output.split()


def launch(command, log_path, *, env, timeout, pid_path=None):
    with Path(log_path).open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        if pid_path is not None:
            write_json(pid_path, {"pid": process.pid, "process_group": process.pid})
        try:
            return process.wait(timeout=timeout), False
        except BaseException as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            return process.returncode, True


def queue_worker(gpu, jobs, events, options, stopping):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(options["cpu_threads"]),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "MLOPS_QUACK_TUNE_DISTRIBUTED": "0",
        "PYTHONUNBUFFERED": "1",
    }
    while not stopping.is_set() and (case := jobs.get()) is not None:
        attempt = Path(case["attempt"])
        attempt.mkdir(parents=True, exist_ok=False)
        for child in ("logs", "nsys"):
            (attempt / child).mkdir()
        begun = time.monotonic()
        events.put({"event": "start", "gpu": gpu, "case": case, "started": stamp()})
        command = [
            options["nsys"],
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--osrt-threshold=10000",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=false",
            "--output=" + str(attempt / "nsys/trace"),
            sys.executable,
            "-u",
            "-m",
            "benchmarking.stage_traces.worker",
            "--artifact",
            case["artifact"],
            "--output",
            str(attempt),
            "--stage",
            case["stage"],
            "--warmup",
            str(options["warmup"]),
            "--iterations",
            str(options["iterations"]),
            "--cpu-threads",
            str(options["cpu_threads"]),
            "--capture",
        ]
        if options["gpu_metrics"]:
            # Nsight resolves the worker's CUDA_VISIBLE_DEVICES mask itself;
            # CUDA ordinal zero need not be physical GPU zero.
            command[2:2] = [
                "--gpu-metrics-devices=cuda-visible",
                f"--gpu-metrics-frequency={options['gpu_metrics_frequency']}",
            ]
        write_json(
            attempt / "command.json",
            {
                "command": command,
                "gpu": gpu,
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "gpu_metrics": options["gpu_metrics"],
                "gpu_metrics_frequency_hz": options["gpu_metrics_frequency"],
                "case": case,
            },
        )
        try:
            code, timed_out = launch(
                command,
                attempt / "logs/console.log",
                env=env,
                timeout=options["timeout"],
                pid_path=attempt / "process.json",
            )
            status = "timeout" if timed_out else "completed" if code == 0 else "failed"
            failure = attempt / "failure.json"
            if failure.exists():
                status = json.loads(failure.read_text())["status"]
            report = attempt / "nsys/trace.nsys-rep"
            result = attempt / "result.json"
            verification = None
            if status == "completed":
                if (
                    not report.is_file()
                    or report.stat().st_size == 0
                    or not result.exists()
                ):
                    raise ValueError("worker exited without a nonempty report/result")
                database = attempt / "nsys/trace.sqlite"
                export = [
                    options["nsys"],
                    "export",
                    "--type=sqlite",
                    "--output=" + str(database),
                    str(report),
                ]
                code, timed_out = launch(
                    export,
                    attempt / "logs/export.log",
                    env=env,
                    timeout=300,
                    pid_path=attempt / "process.json",
                )
                if code or timed_out:
                    raise ValueError("Nsight SQLite export failed; inspect export.log")
                from .report import verify

                device = json.loads(result.read_text())["device_identity"]
                verification = verify(
                    database,
                    options["iterations"],
                    gpu_uuid=device["uuid"] if options["gpu_metrics"] else None,
                )
            outcome = {
                "status": status,
                "returncode": code,
                "gpu": gpu,
                "elapsed_seconds": time.monotonic() - begun,
                "ended": stamp(),
                "case": case,
                "verification": verification,
            }
        except Exception as error:
            outcome = {
                "status": "failed",
                "error": repr(error),
                "gpu": gpu,
                "elapsed_seconds": time.monotonic() - begun,
                "ended": stamp(),
                "case": case,
            }
        write_json(attempt / "outcome.json", outcome)
        events.put({"event": "end", **outcome})


def describe_queue(manifests, output):
    cases = []
    for manifest in manifests:
        for stage in manifest["stages"]:
            for variant in stage["variants"]:
                case = {
                    "stage": stage["name"],
                    "sequence_length": manifest["sequence_length"],
                    "batch_size": manifest["batch_size"],
                    "variant": variant["name"],
                    "artifact": variant["artifact"],
                    "occurrences": stage["occurrences"],
                }
                case["directory"] = str(
                    output
                    / "results"
                    / stage["name"]
                    / f"seqlen-{case['sequence_length']}"
                    / f"batch-{case['batch_size']}"
                    / case["variant"]
                )
                cases.append(case)
    return cases


def run(args):
    args.output = args.output.absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    for child in ("scripts", "logs", "notes", "results", "graphs"):
        (args.output / child).mkdir(exist_ok=True)
    options = {
        k: v
        for k, v in vars(args).items()
        if k not in {"output", "resume", "retry_failed", "discover_only"}
    }
    hashes = {
        str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    manifest_path = args.output / "run-manifest.json"
    manifest = {"options": options, "source_hashes": hashes, "python": sys.executable}
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume requires the same tracing source/settings")
    else:
        write_json(manifest_path, manifest)
    devices = visible_devices(args.gpus)
    if len(set(devices)) != len(devices) or not devices:
        raise ValueError("GPU list must be nonempty and unique")
    captures = []
    for length, batch in itertools.product(args.sequence_lengths, args.batch_sizes):
        directory = args.output / "graphs" / f"seqlen-{length}" / f"batch-{batch}"
        if not (directory / "manifest.json").exists():
            # Preserve a failed discovery rather than treating its directory
            # as an immutable completed inventory.
            if directory.exists():
                directory.rename(
                    directory.with_name(directory.name + f".failed-{time.time_ns()}")
                )
            command = [
                sys.executable,
                "-u",
                "-m",
                "benchmarking.stage_traces.capture",
                "--factory",
                args.factory,
                "--factory-args",
                json.dumps(args.factory_args),
                "--sequence-length",
                str(length),
                "--batch-size",
                str(batch),
                "--output",
                str(directory),
                "--variants",
                *args.variants,
            ]
            print(f"DISCOVER seqlen={length} batch={batch}", flush=True)
            code, timed_out = launch(
                command,
                args.output / "logs" / f"discover-s{length}-b{batch}.log",
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(devices[0])},
                timeout=args.discovery_timeout,
            )
            if code or timed_out:
                raise RuntimeError(
                    f"discovery failed for {length}/{batch}; inspect logs"
                )
        captures.append(json.loads((directory / "manifest.json").read_text()))
    cases = describe_queue(captures, args.output)
    write_json(args.output / "queue.json", {"cases": cases})
    from .report import write_index

    write_index(args.output, cases, [])
    print(
        f"QUEUE {len(cases)} unique stage/variant/geometry traces "
        f"on {len(devices)} GPUs",
        flush=True,
    )
    if args.discover_only:
        return
    pending, finished = [], []
    for index, case in enumerate(cases, 1):
        directory = Path(case["directory"])
        attempts = sorted(directory.glob("attempt-*"))
        prior = attempts[-1] / "outcome.json" if attempts else None
        outcome = json.loads(prior.read_text()) if prior and prior.exists() else None
        if outcome and (outcome["status"] == "completed" or not args.retry_failed):
            finished.append(outcome)
            continue
        pending.append(
            {
                **case,
                "position": index,
                "total": len(cases),
                "attempt": str(directory / f"attempt-{len(attempts) + 1:03d}"),
            }
        )
    # One independent Nsight/Python process per GPU. Lightweight host threads
    # assign work and collect outcomes; they never create a CUDA context.
    jobs, events = queue.Queue(), queue.Queue()
    stopping = threading.Event()
    for case in pending:
        jobs.put(case)
    for _ in devices:
        jobs.put(None)
    workers = [
        threading.Thread(
            target=queue_worker,
            args=(gpu, jobs, events, options, stopping),
            name=f"gpu-{gpu}-controller",
            daemon=True,
        )
        for gpu in devices
    ]
    for worker in workers:
        worker.start()
    active = {}
    ended = 0
    try:
        while ended < len(pending):
            try:
                event = events.get(timeout=30)
            except queue.Empty:
                if all(not worker.is_alive() for worker in workers):
                    raise RuntimeError(
                        f"workers exited before every assignment reported: "
                        f"{[w.name for w in workers]}"
                    ) from None
                continue
            case, gpu = event["case"], str(event["gpu"])
            label = (
                f"{case['stage']} {case['variant']} "
                f"seqlen={case['sequence_length']} batch={case['batch_size']}"
            )
            if event["event"] == "start":
                active[gpu] = case
                print(
                    f"TRACE START [{case['position']}/{len(cases)}] GPU {gpu} {label}\n"
                    f"  log {case['attempt']}/logs/console.log",
                    flush=True,
                )
            else:
                active.pop(gpu, None)
                ended += 1
                finished.append(event)
                write_index(args.output, cases, finished)
                print(
                    f"TRACE {event['status'].upper()} "
                    f"[{case['position']}/{len(cases)}] "
                    f"GPU {gpu} {label} ({event['elapsed_seconds']:.1f}s)",
                    flush=True,
                )
            write_json(
                args.output / "progress.json",
                {
                    "updated": stamp(),
                    "active": active,
                    "finished": finished,
                    "total_cases": len(cases),
                },
            )
    finally:
        stopping.set()
        for case in active.values():
            process_file = Path(case["attempt"]) / "process.json"
            if process_file.exists():
                process_group = json.loads(process_file.read_text())["process_group"]
                with suppress(ProcessLookupError):
                    os.killpg(process_group, signal.SIGTERM)
        for worker in workers:
            worker.join(timeout=2)
    write_index(args.output, cases, finished)
    counts = dict(Counter(v["status"] for v in finished))
    write_json(args.output / "summary.json", {"outcomes": counts, "cases": finished})
    print(f"TRACE SUMMARY {counts}", flush=True)
    if any(key not in {"completed", "out_of_memory"} for key in counts):
        raise RuntimeError("one or more trace cases failed; see per-case outcomes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--factory-args", type=json.loads, default={})
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[8192])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["save", "recompute"],
        default=["save", "recompute"],
    )
    parser.add_argument("--gpus", nargs="+", default=["all"])
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument(
        "--gpu-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record hardware metrics on the worker's assigned GPU (default: on)",
    )
    parser.add_argument("--gpu-metrics-frequency", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--discovery-timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    if (
        min(
            *args.sequence_lengths,
            *args.batch_sizes,
            args.warmup,
            args.iterations,
            args.cpu_threads,
            args.timeout,
        )
        <= 0
    ):
        parser.error("geometry and measurement counts must be positive")
    if not 10 <= args.gpu_metrics_frequency <= 200000:
        parser.error("GPU metric frequency must be between 10 and 200000 Hz")
    run(args)


if __name__ == "__main__":
    main()
