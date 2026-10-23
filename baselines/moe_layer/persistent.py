"""Persistent worker controller with per-case deadlines and OOM recovery."""

import itertools
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from baselines.common import write_json

from .reporting import emit_finished, emit_start


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def group_key(job):
    return (
        job["python"],
        job["ranks"],
        job["distributed_tuning"],
        job["case"].split("/")[0],
    )


def native_allocation_failure(current):
    """Native allocators may print an OOM then hang before Python can raise."""
    if not current or not current["case"].startswith("ncclep/"):
        return None
    path = Path(current["attempt"]) / "console.log"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        text = stream.read().decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if "NCCL EP: failed to allocate" in line]
    if not lines:
        return None
    return {"status": "out_of_memory", "stage": "native_dispatcher_setup",
            "error": lines[0], "source": str(path)}


def run(args, pending, cases):
    known = {case["case"] for case in cases}
    for (python, ranks, distributed_tuning, _backend), grouped in itertools.groupby(
        pending, group_key
    ):
        remaining = list(grouped)
        while remaining:
            workers = args.output / "workers"
            index = len(list(workers.glob("session-*"))) + 1
            session = workers / f"session-{index:03d}"
            session.mkdir(parents=True)
            queue = session / "queue.json"
            write_json(
                queue,
                {
                    "root": str(args.output),
                    "ranks": ranks,
                    "completed_cases": cases,
                    "jobs": remaining,
                },
            )
            command = [python]
            if ranks > 1:
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--max-restarts=0",
                    f"--nproc-per-node={ranks}",
                ]
            command += [
                "-u",
                "-m",
                "baselines.moe_layer.batch_worker",
                "--queue",
                str(queue),
            ]
            # -u belongs to Python, not torchrun's training-script arguments.
            if ranks > 1:
                command.remove("-u")
            write_json(session / "command.json", command)
            current = None
            announced = set()

            def announce(job, announced=announced):
                if job["case"] not in announced:
                    emit_start(args, job["case"], Path(job["attempt"]))
                    announced.add(job["case"])

            begun = time.time()
            timed_out = False
            native_failure = None
            with (session / "console.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "PYTHONHASHSEED": "0",
                        "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": str(args.cpu_threads),
                        "MLOPS_QUACK_TUNE_DISTRIBUTED": "1"
                        if distributed_tuning
                        else "0",
                    },
                )
                try:
                    while True:
                        current = read_json(session / "current.json", current)
                        for job in remaining:
                            outcome = read_json(Path(job["attempt"]) / "outcome.json")
                            if outcome and job["case"] not in known:
                                cases.append({"case": job["case"], **outcome})
                                known.add(job["case"])
                                announce(job)
                                emit_finished(
                                    args, job["case"], Path(job["attempt"]), outcome
                                )
                        if current:
                            announce(current)
                        if process.poll() is not None:
                            break
                        native_failure = native_allocation_failure(current)
                        if native_failure:
                            write_json(Path(current["attempt"]) / "native_failure.json", native_failure)
                            print(f"Native allocation failure; terminating workers: {native_failure['error']}", flush=True)
                            stop(process)
                            break
                        if (
                            time.time() - (current["started"] if current else begun)
                            > args.timeout
                        ):
                            timed_out = True
                            stop(process)
                            break
                        time.sleep(1)
                except BaseException:
                    stop(process)
                    if current and current["case"] not in known:
                        attempt = Path(current["attempt"])
                        outcome = {
                            "status": "interrupted",
                            "attempt": str(attempt.relative_to(args.output)),
                        }
                        write_json(attempt / "outcome.json", outcome)
                        cases.append({"case": current["case"], **outcome})
                    write_json(args.output / "progress.json", {"cases": cases})
                    raise
            # The last completed result may have arrived during process exit.
            uncompleted = []
            for job in remaining:
                outcome = read_json(Path(job["attempt"]) / "outcome.json")
                if outcome and outcome["status"] == "completed":
                    if job["case"] not in known:
                        cases.append({"case": job["case"], **outcome})
                        known.add(job["case"])
                        announce(job)
                        emit_finished(args, job["case"], Path(job["attempt"]), outcome)
                else:
                    uncompleted.append(job)
            if not uncompleted:
                if process.returncode != 0:
                    raise RuntimeError(
                        f"worker teardown failed; inspect {session / 'console.log'}"
                    )
                break
            failed, *remaining = uncompleted
            attempt = Path(failed["attempt"])
            failures = [read_json(path) for path in attempt.glob("rank-*/failure.json")]
            if native_failure:
                failures.append(native_failure)
            status = "timeout" if timed_out else "failed"
            if not timed_out and any(f["status"] == "out_of_memory" for f in failures):
                status = "out_of_memory"
            outcome = {
                "status": status,
                "returncode": process.returncode,
                "failure_stages": sorted({f.get("stage", "unknown") for f in failures}),
                "attempt": str(attempt.relative_to(args.output)),
                "elapsed_seconds": time.time()
                - (current["started"] if current else begun),
                "shared_worker_session": str(session),
            }
            if native_failure:
                outcome["native_failure"] = native_failure
            write_json(attempt / "outcome.json", outcome)
            cases.append({"case": failed["case"], **outcome})
            known.add(failed["case"])
            write_json(args.output / "progress.json", {"cases": cases})
            announce(failed)
            emit_finished(args, failed["case"], attempt, outcome)
            if status != "out_of_memory":
                raise RuntimeError(
                    f"{failed['case']} {status}; "
                    f"inspect {attempt / 'console.log'} and {session / 'console.log'}"
                )
            print("Restarting workers after OOM for the remaining cases", flush=True)
