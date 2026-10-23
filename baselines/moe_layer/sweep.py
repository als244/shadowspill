"""Run selected MoE implementations, preserving every process and sample."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from baselines.common import write_json

from .config import add_shape_arguments, requested_shape
from .implementations.registry import BACKENDS
from .reporting import emit_finished, emit_start


def source_identity():
    import importlib.util

    source = Path(__file__).resolve().parent
    spec = importlib.util.find_spec("mlops")
    if spec is None or spec.origin is None:
        raise RuntimeError("MLOps is not importable")
    mlops = Path(spec.origin).parent
    return {
        prefix + "/" + str(p.relative_to(root)): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for prefix, root in (("benchmark", source), ("mlops", mlops))
        for p in sorted(root.rglob("*.py"))
    }


def selected_result(case):
    attempts = sorted(case.glob("attempt-*"))
    if not attempts:
        return None
    path = attempts[-1] / "outcome.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "interrupted"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 65536, 131072],
    )
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument(
        "--nvidia-python",
        default=str(
            Path(sys.executable).resolve().parents[2] / "nvidia-training/bin/python"
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=list(BACKENDS),
        default=["sonicmoe", "quackmoe"],
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=["bf16", "fp8_current"],
        default=["bf16", "fp8_current"],
    )
    parser.add_argument(
        "--training", nargs="+", choices=["full", "lora"], default=["full", "lora"]
    )
    parser.add_argument(
        "--distributed-tuning",
        action="store_true",
        help="share Quack candidate searches across lockstep DP ranks",
    )
    parser.add_argument(
        "--fresh-workers",
        action="store_true",
        help="disable shared process/model setup between cases",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--sonicmoe-candidate-policy",
        choices=["baseline", "expanded", "expanded-k"],
        default="baseline",
    )
    add_shape_arguments(parser)
    args = parser.parse_args(argv)
    if min(*args.tokens, args.ep_size, args.warmup, args.iterations, args.timeout) < 1:
        parser.error("counts and timeout must be positive")
    shape = requested_shape(args, args.ep_size)
    if shape.experts % args.ep_size:
        parser.error("EP group must divide the expert count")
    if args.dp_size < 1:
        parser.error("DP size must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        options = {
            k: v
            for k, v in vars(args).items()
            if k not in {"output", "resume", "retry_failed", "dry_run"}
        }
        manifest = {
            "schema_version": 1,
            "options": options,
            "source_hashes": source_identity(),
            "python": sys.executable,
        }
        saved = args.output / "manifest.json"
        if saved.exists():
            if not args.resume or json.loads(saved.read_text()) != manifest:
                raise ValueError(
                    "resume requires identical source/configuration; "
                    "use a new output for changes"
                )
        else:
            write_json(saved, manifest)
        cases = []
        pending = []
        for backend, precision, training, tokens in itertools.product(
            args.backends, args.precisions, args.training, args.tokens
        ):
            name = f"{backend}/{precision}/{training}/tokens-{tokens}"
            case = args.output / "cases" / name
            previous = selected_result(case)
            if previous and (
                previous["status"] == "completed" or not args.retry_failed
            ):
                cases.append({"case": name, **previous})
                continue
            attempts = sorted(case.glob("attempt-*"))
            attempt = case / f"attempt-{len(attempts) + 1:03d}"
            entry = BACKENDS[backend]
            interpreter = (
                args.nvidia_python
                if entry.environment == "nvidia-training"
                else sys.executable
            )
            ranks = args.ep_size if entry.parallelism == "ep" else args.dp_size
            command = [interpreter]
            if ranks > 1:
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--max-restarts=0",
                    f"--nproc-per-node={ranks}",
                ]
            command += [
                "-m",
                "baselines.moe_layer.worker",
                "--backend",
                backend,
                "--precision",
                precision,
                "--training",
                training,
                "--tokens",
                str(tokens),
                "--output",
                str(attempt),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--cpu-threads",
                str(args.cpu_threads),
                "--seed",
                str(args.seed),
                "--lora-rank",
                str(args.lora_rank),
                "--sonicmoe-candidate-policy",
                args.sonicmoe_candidate_policy,
            ]
            command.extend(
                [
                    "--model",
                    args.model,
                    "--lora-alpha",
                    str(shape.lora_alpha),
                    "--router-weight-dtype",
                    args.router_weight_dtype,
                ]
            )
            for field in ("width", "hidden", "experts", "top_k"):
                command.extend(
                    ["--" + field.replace("_", "-"), str(getattr(shape, field))]
                )
            if args.tiny:
                command.append("--tiny")
            if args.dry_run:
                cases.append({"case": name, "status": "prepared", "command": command})
                continue
            if not args.fresh_workers:
                module_at = command.index("baselines.moe_layer.worker")
                pending.append(
                    {
                        "case": name,
                        "attempt": str(attempt),
                        "arguments": command[module_at + 1 :],
                        "command": command,
                        "python": interpreter,
                        "ranks": ranks,
                        "distributed_tuning": args.distributed_tuning
                        and entry.parallelism == "dp"
                        and ranks > 1,
                    }
                )
                continue
            attempt.mkdir(parents=True)
            write_json(attempt / "command.json", command)
            emit_start(args, name, attempt)
            begun = time.time()
            with (attempt / "console.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "PYTHONHASHSEED": "0",
                        "OMP_NUM_THREADS": str(args.cpu_threads),
                        "MLOPS_QUACK_TUNE_DISTRIBUTED": (
                            "1"
                            if args.distributed_tuning
                            and entry.parallelism == "dp"
                            and ranks > 1
                            else "0"
                        ),
                    },
                )
                try:
                    returncode = process.wait(timeout=args.timeout)
                    status = "completed" if returncode == 0 else "failed"
                except BaseException as error:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    status = (
                        "timeout"
                        if isinstance(error, subprocess.TimeoutExpired)
                        else "interrupted"
                    )
                    if status == "interrupted":
                        write_json(attempt / "outcome.json", {"status": status})
                        raise
                    returncode = process.returncode
            result_path = attempt / "result.json"
            result = (
                json.loads(result_path.read_text()) if result_path.exists() else None
            )
            expected_ranks = ranks
            if status == "completed" and (
                result is None or result.get("all_ranks_validated") != expected_ranks
            ):
                status = "invalid_result"
            failures = [
                json.loads(p.read_text()) for p in attempt.glob("rank-*/failure.json")
            ]
            if any(f["status"] == "out_of_memory" for f in failures):
                status = "out_of_memory"
            outcome = {
                "status": status,
                "returncode": returncode,
                "failure_stages": sorted({f.get("stage", "unknown") for f in failures}),
                "attempt": str(attempt.relative_to(args.output)),
                "elapsed_seconds": time.time() - begun,
            }
            write_json(attempt / "outcome.json", outcome)
            cases.append({"case": name, **outcome})
            write_json(args.output / "progress.json", {"cases": cases})
            emit_finished(args, name, attempt, outcome)
            if status not in {"completed", "out_of_memory"}:
                raise RuntimeError(
                    f"{name} {status}; inspect {attempt / 'console.log'}"
                )
        if pending:
            from .persistent import run

            run(args, pending, cases)
        write_json(args.output / "progress.json", {"cases": cases})


if __name__ == "__main__":
    main()
