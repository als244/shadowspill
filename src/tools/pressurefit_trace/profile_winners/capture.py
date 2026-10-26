"""One process-tree Nsight report per saved winner, with checked resumability."""

import re
import subprocess

from benchmarking.processes import worker_command
from benchmarking.quickstart.phases.common import write

from .evidence import validate_sqlite
from .selection import completed, digest, fingerprint, verified_capture


def supported_traces(nsys):
    help_text = subprocess.check_output([nsys, "profile", "--help"], text=True)
    return "cuda,nvtx,osrt" + (",nccl" if re.search(r"\bnccl\b", help_text) else "")


def command(args, report, budget, world, *, plan_digest, traces="cuda,nvtx,osrt"):
    return [
        args.nsys,
        "profile",
        f"--trace={traces}",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-memory-usage=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--flush-on-cudaprofilerstop=false",
        "--gpu-metrics-devices=cuda-visible",
        f"--gpu-metrics-frequency={args.gpu_metrics_frequency}",
        "--wait=all",
        "--output",
        str(report),
        *worker_command(
            "tools.pressurefit_trace.profile_winners",
            [
                args.quickstart,
                "--worker",
                "--output",
                report.parent / "execution",
                "--budget-gib",
                budget,
                "--world-size",
                world,
                "--plan-digest",
                plan_digest,
                "--warmup-steps",
                args.warmup_steps,
                "--capture-steps",
                args.capture_steps,
                "--capture-settle-seconds",
                args.capture_settle_seconds,
            ],
            ranks=world,
        ),
    ]


def run_logged(argv, log_path):
    write(log_path.with_suffix(".command.json"), argv)
    with log_path.open("w") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)


def capture(args):
    request, index, selected = completed(
        args.quickstart, budget_gib=args.budget_gib, world_size=args.world_size
    )
    world = index.get("world_size", 1)
    traces = supported_traces(args.nsys)
    for winner in selected:
        budget = winner["budget"] / (1 << 30)
        case = args.output / f"budget_{budget:g}GiB"
        inputs = fingerprint(args.quickstart, request, index, winner, args)
        inputs["trace_types"] = traces
        if verified_capture(case, inputs):
            print(f"capture: verified existing {case}", flush=True)
            continue
        number = 1
        while (case / f"attempt-{number:03d}").exists():
            number += 1
        attempt = case / f"attempt-{number:03d}"
        attempt.mkdir(parents=True)
        report = attempt / "winner"
        print(
            f"capture: {budget:g} GiB, {world} workers; log {attempt / 'capture.log'}",
            flush=True,
        )
        run_logged(
            command(
                args,
                report,
                budget,
                world,
                plan_digest=winner["plan_digest"],
                traces=traces,
            ),
            attempt / "capture.log",
        )
        raw, exported = report.with_suffix(".nsys-rep"), report.with_suffix(".sqlite")
        run_logged(
            [args.nsys, "export", "--type=sqlite", "--output", str(exported), str(raw)],
            attempt / "export.log",
        )
        validation = validate_sqlite(
            exported,
            world_size=world,
            capture_steps=args.capture_steps,
            warmup_steps=args.warmup_steps,
        )
        write(
            case / "validation.json",
            {
                **validation,
                "report": str(raw.relative_to(case)),
                "sqlite": str(exported.relative_to(case)),
                "physical_budget_gib": budget,
                "performance_result": False,
                "inputs": inputs,
                "report_sha256": digest(raw),
                "sqlite_sha256": digest(exported),
            },
        )
        print(f"capture: validated {raw}", flush=True)
