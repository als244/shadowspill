"""Separate Nsight Systems captures of saved, already benchmarked winners.

Run on the allocated GPU node with the original source/native environment:
python -m tools.pressurefit_trace.profile_winners QUICKSTART --output OUTPUT
Each budget gets one process-tree report, including all ranks and GPU metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def replay_request_resolver(resolve, recorded_request):
    """Preserve admission's original cache policy while keeping replay guards.

    Quickstart's --reproduce changes plan-store policy to require. Distributed
    admission can need a new local cache entry even with the saved incumbent;
    use the original policy and verify the final plan before starting capture.
    """

    def restored(parser, arguments):
        arguments.plan_store_mode = recorded_request["plan_store_mode"]
        return resolve(parser, arguments)

    return restored


def require_exact_plan(actual, expected):
    for key in (
        "program_digest",
        "schedule_digest",
        "layout_digest",
        "capture_identity",
    ):
        if actual[key] != expected[key]:
            raise ValueError(
                f"Nsight replay changed {key}: {actual[key]} != {expected[key]}"
            )


def execution_structure(program):
    """Exclude measured costs while retaining every storage/operation contract."""
    value = copy.deepcopy(program)
    for profile in value["profiles"]:
        for field in ("runtime_ns", "compatibility_digest", "overlap_costs"):
            profile.pop(field, None)
        for phase in profile.get("resource_phases", []):
            phase.pop("duration_ns", None)
    for alias in value["alias_groups"]:
        transfer = alias.get("transfer")
        if transfer is not None:
            for direction in ("fetch", "evict"):
                for phase in transfer[direction]:
                    phase.pop("duration_ns", None)
    return value


def restore_replay_program(programs, saved):
    """Bind current compiled storage contracts to the saved timing evidence.

    Profiling a process may select a different timing cache entry. An exact
    replay uses the original costs and certificate, after checking that all
    tasks, objects, communication operations, and allocation facts still match.
    No runtime communication or allocation check is disabled.
    """
    from dataclasses import replace

    from shadowspill.pipeline.admission import dynamic_scratch_reserve_bytes

    current = programs.lowered.program
    expected = saved.result.program
    if execution_structure(current.to_dict()) != execution_structure(
        expected.to_dict()
    ):
        raise ValueError(
            "Nsight replay changed task, storage, or communication structure"
        )
    if programs.lowered.initial_residency != saved.result.initial_residency:
        raise ValueError("Nsight replay changed initial residency")
    if programs.lowered.final_residency != saved.result.final_residency:
        raise ValueError("Nsight replay changed final residency")
    facts = saved.effective_facts
    if programs.admission.object_capacity_bytes < facts.object_capacity_bytes:
        raise ValueError("Nsight replay leaves insufficient object capacity")
    if (
        replace(programs.admission, object_capacity_bytes=facts.object_capacity_bytes)
        != facts
    ):
        raise ValueError(
            "Nsight replay changed compiled allocation contracts or pool capacity"
        )
    scratch = dynamic_scratch_reserve_bytes(
        programs.measurements_by_profile,
        minimum_bytes=programs.dynamic_scratch_reserve_bytes,
    )
    if scratch > saved.fixed_layout.scratch_reserve_bytes:
        raise ValueError(
            "Nsight replay requires more dynamic scratch than the saved plan"
        )
    # Profile cache keys include measurement context. The storage facts above
    # verify their allocation contracts; preserve current compiled envelopes
    # under the original program's profile keys for normal runtime admission.
    measurements = dict(programs.measurements_by_profile)
    current_profiles = {profile.profile_id: profile for profile in current.profiles}
    for profile in expected.profiles:
        key = current_profiles[profile.profile_id].compatibility_digest
        if key in programs.measurements_by_profile:
            measurements[profile.compatibility_digest] = (
                programs.measurements_by_profile[key]
            )
    return replace(
        programs,
        lowered=replace(programs.lowered, program=expected),
        measurements_by_profile=measurements,
    )


def replay_selection(saved):
    """Use the exact admitted winner, including its original physical layout."""
    from shadowspill.planner.admission.layout import FixedLayoutAdmission
    from shadowspill.planner.admission.refinement import FixedLayoutSelection
    from shadowspill.planner.plan_store import PlanLookup

    certificate = FixedLayoutAdmission(
        saved.fixed_layout, saved.simulation_admission, saved.simulation
    )
    return FixedLayoutSelection(
        plan=PlanLookup(saved.result, from_store=True, certificate=certificate),
        facts=saved.effective_facts,
        admission=certificate,
        attempts=saved.attempts,
        original_object_capacity_bytes=saved.program.admission_facts.object_capacity_bytes,
    )


def validate_sqlite(path, *, world_size, capture_steps=None, warmup_steps=3):
    """Require actual CUDA, NVTX and device-metric rows, not only an exit code."""
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as database:
        tables = {
            r[0]
            for r in database.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        counts = {}
        for name in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "NVTX_EVENTS",
            "GPU_METRICS",
        ):
            counts[name] = (
                database.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                if name in tables
                else 0
            )
        if any(count == 0 for count in counts.values()):
            raise ValueError(f"Nsight capture lacks requested evidence: {counts}")
        columns = {
            row[1] for row in database.execute('PRAGMA table_info("GPU_METRICS")')
        }
        field = (
            "typeId" if "typeId" in columns else "gpuId" if "gpuId" in columns else None
        )
        if field is None:
            raise ValueError(
                "unknown GPU metric schema; inspect GPU_METRICS before accepting"
            )
        groups = list(
            database.execute(
                f'SELECT "{field}", count(*) FROM GPU_METRICS GROUP BY "{field}"'
            )
        )
        if len(groups) < world_size:
            raise ValueError(f"metrics do not cover all {world_size} GPUs: {groups}")
        if "TARGET_INFO_CUDA_DEVICE" in tables:
            # CUDA ordinals are process-local: rank_device exposes each rank's
            # physical GPU as cuda:0. Resolve the process/ordinal pair before
            # checking physical GPU coverage. Nsight packs PID into bits 24:48.
            kernel_devices = list(
                database.execute(
                    "SELECT d.gpuId, count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    "LEFT JOIN TARGET_INFO_CUDA_DEVICE d "
                    "ON d.pid=((k.globalPid >> 24) & 0xffffff) "
                    "AND d.cudaId=k.deviceId GROUP BY d.gpuId"
                )
            )
            if any(device is None for device, _ in kernel_devices):
                raise ValueError("cannot map CUDA kernels to physical GPUs")
        else:
            kernel_devices = list(
                database.execute(
                    "SELECT deviceId, count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                    "GROUP BY deviceId"
                )
            )
        if len(kernel_devices) < world_size:
            raise ValueError(
                f"CUDA kernels do not cover all {world_size} GPUs: {kernel_devices}"
            )
        text_columns = {
            row[1] for row in database.execute('PRAGMA table_info("NVTX_EVENTS")')
        }
        if (
            "text" not in text_columns
            or "textId" not in text_columns
            or "StringIds" not in tables
        ):
            raise ValueError("unknown NVTX text schema; inspect before accepting")
        labels = dict(
            database.execute(
                "SELECT coalesce(n.text, s.value), count(*) FROM NVTX_EVENTS n "
                "LEFT JOIN StringIds s ON n.textId=s.id "
                "GROUP BY coalesce(n.text,s.value)"
            )
        )
        task_ranges = sum(
            count
            for name, count in labels.items()
            if name and name.startswith("shadowspill.pytorch.task.")
        )
        transfer_ranges = sum(
            count
            for name, count in labels.items()
            if name and name.startswith("shadowspill.runtime.transfer.")
        )
        steps = {
            name: count
            for name, count in labels.items()
            if name and name.startswith("shadowspill.capture/")
        }
        step_ranks = {
            int(found.group(1))
            for name in steps
            if (found := re.search(r"/rank-(\d+)/step-", name)) is not None
        }
        if (
            not task_ranges
            or not transfer_ranges
            or step_ranks != set(range(world_size))
        ):
            raise ValueError(
                "Nsight capture lacks semantic task/transfer/all-rank step ranges"
            )
        if capture_steps is not None:
            observed = {
                (int(match.group(1)), int(match.group(2)))
                for name, count in steps.items()
                if count == 1 and (match := re.search(r"/rank-(\d+)/step-(\d+)$", name))
            }
            expected = {
                (rank, step)
                for rank in range(world_size)
                for step in range(warmup_steps + 1, warmup_steps + capture_steps + 1)
            }
            if observed != expected:
                raise ValueError("Nsight capture does not contain every requested step")
        return {
            "status": "validated",
            "rows": counts,
            "metric_groups": groups,
            "kernel_devices": kernel_devices,
            "task_ranges": task_ranges,
            "transfer_ranges": transfer_ranges,
            "step_ranges": steps,
        }


def worker(args):
    """Reuse baseline archives and reject any change in the admitted schedule."""
    import torch
    import torch.distributed as dist

    from benchmarking.quickstart import cli, runner
    from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
    from shadowspill.pytorch import StepSearchReport
    from shadowspill.pytorch.planning.training import build
    from shadowspill.step import StepDataOrdering

    rank = int(os.environ["RANK"])
    source = args.quickstart / f"rank-{rank:05d}"
    records = [
        json.loads(p.read_text()) for p in (source / "executed").glob("*bytes.json")
    ]
    expected = next(
        r
        for r in records
        if args.budget_gib * 2**30 in r["requested_execution_budget_bytes"]
    )
    if expected["status"] != "executed":
        raise ValueError("cannot profile a winner that did not complete the benchmark")
    budget = expected["execution_budget_bytes"]
    bundle = source / "winners" / f"{budget}bytes"
    identity = json.loads((bundle / "bundle.json").read_text())
    search_plan = source / "search-winners" / f"{budget}bytes.plan.json"
    incumbent = AnnotatedProgramPlan.from_json(search_plan.read_text())
    saved = AnnotatedProgramPlan.from_json((bundle / "plan.json").read_text())
    original_program_builder = build.build_training_programs

    def exact_program_builder(*values, **options):
        programs = original_program_builder(*values, **options)
        restored = restore_replay_program(programs, saved)
        write(
            args.output / f"rank-{rank:05d}" / "replay-contract.json",
            {
                "status": "matched_storage_and_operations",
                "source_plan": str(bundle / "plan.json"),
                "current_profile_program_digest": programs.lowered.program.digest,
                "replayed_program_digest": restored.lowered.program.digest,
                "timing_source": "original benchmark winner",
            },
        )
        return restored

    build.build_training_programs = exact_program_builder
    build.plan_training_programs = lambda programs, **options: replay_selection(saved)
    original_steps = runner.Tour._run_steps
    recorded_request = json.loads((source / "request.json").read_text())["request"]
    cli.resolve_request = replay_request_resolver(cli.resolve_request, recorded_request)

    def budgets(runtime, request):
        if budget > runtime.pools["execution"].capacity:
            raise ValueError(
                "profiler leaves insufficient capacity for the exact baseline plan"
            )
        physical = expected["requested_execution_budget_bytes"][0]
        return runner.Budgets([physical], [physical], {physical: budget})

    def search(self):
        self.report = StepSearchReport.load(source / "search.json")
        self.report.save(self.paths.root / "search.json")
        self.plan_log.note("Nsight replay: use saved winner; no geometry/policy search")

    def run(self):
        outcome = self.run_one_budget(
            budget,
            expected["candidate"],
            StepDataOrdering.from_dict(expected["ordering"], "ordering"),
            incumbent=incumbent,
        )
        return [] if outcome is None else [outcome]

    def steps(self, training, plan_report, candidate, *, budget):
        actual = {
            "program_digest": plan_report.program.digest,
            "schedule_digest": plan_report.execution_plan.schedule.digest,
            "layout_digest": plan_report.physical_layout.digest,
            "capture_identity": plan_report.capture_identity,
        }
        require_exact_plan(actual, identity)
        write(
            self.paths.root / "exact-plan.json",
            {
                **actual,
                "status": "matched",
                "source": str(bundle),
                "physical_budget_gib": args.budget_gib,
                "execution_budget_bytes": budget,
                "warmup_steps": args.warmup_steps,
                "capture_steps": args.capture_steps,
                "capture_settle_seconds": args.capture_settle_seconds,
                "performance_result": False,
            },
        )
        # Profiler stop drains CUDA activity across the process tree. A rank
        # entering a CUDA/NCCL barrier while another rank is stopping capture
        # can leave that drain waiting for a kernel that needs the stopped rank.
        # Coordinate capture boundaries on CPU instead; no GPU work is queued
        # until every rank has returned from the profiler API.
        capture_control = dist.new_group(backend="gloo")

        class Annotated:
            def __init__(self):
                self.step = 0
                self.active = False

            def __getattr__(self, name):
                return getattr(training, name)

            def __call__(self, inputs, **kwargs):
                self.step += 1
                if self.step == args.warmup_steps + 1:
                    training.synchronize()
                    torch.cuda.nvtx.range_push("shadowspill.capture.initialize")
                    torch.cuda.nvtx.range_pop()
                    dist.barrier(group=capture_control)
                    if rank == 0:
                        torch.cuda.cudart().cudaProfilerStart()
                    dist.barrier(group=capture_control)
                    # Nsight enables CPU/NVTX collection asynchronously in
                    # sibling processes. Keep this settling interval outside
                    # every annotated training step, on both capture edges.
                    time.sleep(args.capture_settle_seconds)
                    dist.barrier(group=capture_control)
                    self.active = True
                kwargs["profiler_annotations"] = self.active
                if self.active:
                    torch.cuda.nvtx.range_push(
                        f"shadowspill.capture/budget-{args.budget_gib:g}GiB/rank-{rank}/step-{self.step}"
                    )
                return training(inputs, **kwargs)

            def synchronize(self):
                training.synchronize()
                if self.active:
                    torch.cuda.nvtx.range_pop()
                    if self.step == args.warmup_steps + args.capture_steps:
                        dist.barrier(group=capture_control)
                        time.sleep(args.capture_settle_seconds)
                        dist.barrier(group=capture_control)
                        if rank == 0:
                            torch.cuda.cudart().cudaProfilerStop()
                        dist.barrier(group=capture_control)
                        self.active = False

        previous = self.arguments.steps
        self.arguments.steps = args.warmup_steps + args.capture_steps + 1
        try:
            return original_steps(
                self, Annotated(), plan_report, candidate, budget=budget
            )
        finally:
            self.arguments.steps = previous

    cli.plan_budgets = budgets
    runner.Tour.search = search
    runner.Tour.run = run
    runner.Tour._run_steps = steps
    sys.argv = [
        "quickstart",
        "--reproduce",
        str(args.quickstart),
        "--output-dir",
        str(args.output),
    ]
    return cli.main()


def capture(args):
    source = args.quickstart / "rank-00000"
    records = [
        json.loads(p.read_text()) for p in (source / "executed").glob("*bytes.json")
    ]
    if not records:
        raise ValueError("baseline has no executed winners")
    budgets = sorted(
        r["requested_execution_budget_bytes"][0] / 2**30
        for r in records
        if r["status"] == "executed"
    )
    if args.budget_gib is not None:
        if args.budget_gib not in budgets:
            raise ValueError("requested budget has no completed winner")
        budgets = [args.budget_gib]
    for budget in budgets:
        case = args.output / f"budget_{budget:g}GiB"
        case.mkdir(parents=True, exist_ok=True)
        inputs = {}
        for rank in range(args.world_size):
            rank_source = args.quickstart / f"rank-{rank:05d}"
            execution = next(
                json.loads(p.read_text())
                for p in (rank_source / "executed").glob("*bytes.json")
                if budget * 2**30
                in json.loads(p.read_text())["requested_execution_budget_bytes"]
            )
            logical = execution["execution_budget_bytes"]
            bundle = rank_source / "winners" / f"{logical}bytes" / "bundle.json"
            inputs[str(rank)] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        fingerprint = {
            "bundles": inputs,
            "warmup_steps": args.warmup_steps,
            "capture_steps": args.capture_steps,
            "capture_settle_seconds": args.capture_settle_seconds,
            "gpu_metrics_frequency": args.gpu_metrics_frequency,
            "capture_protocol": "cpu_barriers_settle_no_stop_flush_v2",
        }
        if (case / "validation.json").exists():
            previous = json.loads((case / "validation.json").read_text())
            if previous.get("inputs") != fingerprint:
                raise ValueError(f"existing capture has different inputs: {case}")
            for key in ("report", "sqlite"):
                with (case / previous[key]).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != previous[f"{key}_sha256"]:
                    raise ValueError(f"saved capture changed: {case / previous[key]}")
            continue
        # Never overwrite a failed/partial capture with a new attempt.
        index = 1
        while (case / f"attempt-{index:03d}").exists():
            index += 1
        attempt = case / f"attempt-{index:03d}"
        attempt.mkdir()
        report = attempt / "winner"
        command = [
            args.nsys,
            "profile",
            "--trace=cuda,nvtx,osrt,nccl",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-memory-usage=true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            # Avoid Nsight's synchronous cross-context flush inside the CUDA
            # profiler API. Collection-end still flushes the trace buffers.
            "--flush-on-cudaprofilerstop=false",
            "--gpu-metrics-devices=all",
            f"--gpu-metrics-frequency={args.gpu_metrics_frequency}",
            "--wait=all",
            "--output",
            str(report),
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.world_size}",
            "-m",
            "benchmarking.rank_device",
            str(Path(__file__).resolve()),
            str(args.quickstart),
            "--worker",
            "--output",
            str(attempt / "quickstart"),
            "--budget-gib",
            str(budget),
            "--warmup-steps",
            str(args.warmup_steps),
            "--capture-steps",
            str(args.capture_steps),
            "--capture-settle-seconds",
            str(args.capture_settle_seconds),
        ]
        write(attempt / "command.json", command)
        with (attempt / "capture.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        raw = report.with_suffix(".nsys-rep")
        exported = report.with_suffix(".sqlite")
        with (attempt / "export.log").open("w") as log:
            subprocess.run(
                [
                    args.nsys,
                    "export",
                    "--type=sqlite",
                    "--output",
                    str(exported),
                    str(raw),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        validation = validate_sqlite(
            exported,
            world_size=args.world_size,
            capture_steps=args.capture_steps,
            warmup_steps=args.warmup_steps,
        )
        validation.update(
            report=str(raw.relative_to(case)),
            sqlite=str(exported.relative_to(case)),
            physical_budget_gib=budget,
            performance_result=False,
            inputs=fingerprint,
        )
        for key, path in (("report", raw), ("sqlite", exported)):
            with path.open("rb") as stream:
                validation[f"{key}_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        write(case / "validation.json", validation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quickstart", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--capture-steps", type=int, default=3)
    parser.add_argument(
        "--capture-settle-seconds",
        type=float,
        default=1.0,
        help="idle interval at each capture edge, outside annotated training steps",
    )
    parser.add_argument("--gpu-metrics-frequency", type=int, default=10000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.quickstart, args.output = args.quickstart.resolve(), args.output.resolve()
    if min(args.world_size, args.warmup_steps, args.capture_steps) < 1:
        parser.error("world size and step counts must be positive")
    if (
        not math.isfinite(args.capture_settle_seconds)
        or args.capture_settle_seconds < 0
    ):
        parser.error("capture settling interval must be finite and nonnegative")
    return worker(args) if args.worker else capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
