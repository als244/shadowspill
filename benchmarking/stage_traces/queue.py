"""Resume trace attempts and assign independent cases to one worker per GPU."""

import json
import queue
import threading
from collections import Counter
from pathlib import Path

from benchmarking._serialization import atomic_json

from .nsys import trace_case
from .process import signal_group, stamp
from .report import write_index


def select_attempts(cases, *, retry_failed):
    pending, finished = [], []
    for index, case in enumerate(cases, 1):
        directory = Path(case["directory"])
        attempts = sorted(directory.glob("attempt-*"))
        prior = attempts[-1] / "outcome.json" if attempts else None
        outcome = json.loads(prior.read_text()) if prior and prior.exists() else None
        if outcome and (outcome["status"] == "completed" or not retry_failed):
            finished.append(outcome)
        else:
            pending.append(
                {
                    **case,
                    "position": index,
                    "total": len(cases),
                    "attempt": str(directory / f"attempt-{len(attempts) + 1:03d}"),
                }
            )
    return pending, finished


def queue_worker(gpu, jobs, events, options, stopping):
    while not stopping.is_set() and (case := jobs.get()) is not None:
        events.put({"event": "start", "gpu": gpu, "case": case, "started": stamp()})
        events.put({"event": "end", **trace_case(case, gpu, options)})


def record_event(event, active, finished, cases, output):
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
        finished.append(event)
        write_index(output, cases, finished)
        print(
            f"TRACE {event['status'].upper()} [{case['position']}/{len(cases)}] "
            f"GPU {gpu} {label} ({event['elapsed_seconds']:.1f}s)",
            flush=True,
        )
    atomic_json(
        output / "progress.json",
        {
            "updated": stamp(),
            "active": active,
            "finished": finished,
            "total_cases": len(cases),
        },
    )


def run_queue(cases, devices, options, output, *, retry_failed=False):
    pending, finished = select_attempts(cases, retry_failed=retry_failed)
    jobs, events = queue.Queue(), queue.Queue()
    stopping = threading.Event()
    for case in pending:
        jobs.put(case)
    for _ in devices:
        jobs.put(None)
    # Host threads assign work; only fresh Nsight children create CUDA contexts.
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
    try:
        while len(finished) < len(cases):
            try:
                event = events.get(timeout=30)
            except queue.Empty:
                if all(not worker.is_alive() for worker in workers):
                    raise RuntimeError(
                        "workers exited before every assignment reported"
                    ) from None
                continue
            record_event(event, active, finished, cases, output)
    finally:
        stopping.set()
        for case in active.values():
            path = Path(case["attempt"]) / "process.json"
            if path.exists():
                signal_group(json.loads(path.read_text())["process_group"])
        for worker in workers:
            worker.join(timeout=2)
    write_index(output, cases, finished)
    counts = dict(Counter(row["status"] for row in finished))
    atomic_json(output / "summary.json", {"outcomes": counts, "cases": finished})
    print(f"TRACE SUMMARY {counts}", flush=True)
    if any(key not in {"completed", "out_of_memory"} for key in counts):
        raise RuntimeError("one or more trace cases failed; see per-case outcomes")
