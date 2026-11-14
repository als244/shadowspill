"""Bounded-phase FP16 DP qualification through the public saved-Plan path."""

import json
import os
import sys
from pathlib import Path


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value, indent=2, default=dict) + "\n")


def build(directory, case):
    from collections.abc import Mapping

    import torch
    from torch._inductor import config

    from engine_tests.execution._distributed_regions import (
        check_batch_chunk_metadata,
        objective,
        policies,
    )
    from engine_tests.execution.narrow_regions.fixture import model_for
    from shadowspill import build_forward_backward, build_step

    reference = torch.load(directory / "reference.pt", weights_only=True)
    model = model_for(case, device="meta")
    options = policies(case)
    common = dict(
        objective=objective,
        example_calls=reference["examples"][0][0],
        grad_dtype=torch.float16,
        fuse_narrow_gradient_accumulation=True,
        **options,
    )
    if case["kind"] == "step":
        common.update(
            optimizer=torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=0.001,
                foreach=False,
            ),
            master_dtype=torch.float32,
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    with config.patch(emulate_precision_casts=False):
        program = (build_step if case["kind"] == "step" else build_forward_backward)(
            model, **common
        )
    geometry = program.bindings["gradient_geometry"]
    assert geometry and {g["dtype"] for g in geometry.values()} == {"torch.float16"}
    if case["kind"] == "step":
        bindings = {v["name"]: v for v in program.bindings["optimizer_bindings"]}
        for name in reference["trainable"]:
            assert bindings[name]["dtype"] == "torch.float32", name
            assert bindings["compute." + name]["dtype"] == "torch.float16", name
        for name, item in bindings.items():
            if name.endswith((".exp_avg", ".exp_avg_sq")):
                assert item["dtype"] == "torch.float32", name
    chunks, nested = [], []

    def walk(value):
        if isinstance(value, Mapping):
            if "row_chunks" in value:
                chunks.append(value["row_chunks"])
            if "child_regions" in value:
                nested.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                walk(child)

    walk(program.bindings["task_metadata"])
    assert {3, 7, 11} <= {c["chunk_size"] for c in chunks}
    assert nested
    if case.get("chunk_axis") == "batch":
        check_batch_chunk_metadata(program.bindings["task_metadata"])
    selector = options["chunking"].lifetimes.select
    assert selector.calls > 0
    selector.building = False
    program.save(directory / "program.ssp")
    write(
        directory,
        "built.json",
        dict(
            dtype="float16",
            gradient_dtype="float16",
            master_dtype="float32" if case["kind"] == "step" else None,
            emulate_precision_casts=False,
            fuse_narrow_gradient_accumulation=True,
            tasks=len(program.tasks),
            executable_keys=len(program.executables),
            lifetime_selector_calls=selector.calls,
            nested_compositions=len(nested),
            chunk_sizes=sorted({c["chunk_size"] for c in chunks}),
            graph_pair_policy=options["graph_pair_policy"].identity(),
            parallelism=options["parallelism"].to_dict(),
            setup=dict(program.setup),
        ),
    )


def profile(directory, case):
    from shadowspill import Program, apply_profiles, measure_program

    program = Program.load(directory / "program.ssp")
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.003)
    profiles.save(directory / "profiles.ssp")
    measured = apply_profiles(program, profiles)
    assert not measured.estimated_memory
    measured.save(directory / "profiled.ssp")


def communicate(directory, case):
    from datetime import timedelta

    import torch.distributed as dist

    from shadowspill import Program, measure_communication
    from shadowspill.memory import pinned_host

    dist.init_process_group("gloo", timeout=timedelta(seconds=75))
    try:
        info = measure_communication(
            program=Program.load(directory / "program.ssp"),
            device="cuda:0",
            rank_map={0: 1, 1: 0},
            sizes_bytes=(1024, 1 << 20, 8 << 20),
            warmup_seconds=0.003,
            measure_seconds=0.006,
            spill=pinned_host(capacity=64 << 20),
        )
        if dist.get_rank() == 0:
            info.save(directory / "communication.ssp")
    finally:
        dist.destroy_process_group()


def plan(directory, case):
    from engine_tests.execution._distributed_phases import plan as ordinary

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    ordinary(directory, case["variant"])
    assert "torch" not in sys.modules


def execute(directory, case, *, resume=False):
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import triton.runtime.autotuner
    import triton.runtime.jit

    from engine_tests.execution._cold import forbid_preparation, forbidden
    from engine_tests.execution.narrow_regions.state import export, same_pieces
    from qualification.runtime_evidence import (
        adapter_statistics,
        check_physical_budget,
        statistics_dict,
    )
    from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint
    from shadowspill.memory import pinned_host
    from shadowspill.runtime.occupancy import live_allocations

    selected = Plan.load(directory / "plan.ssp")
    assert not selected.simulation_only_reasons
    forbid_preparation(
        "engine_tests.execution.narrow_regions.fixture",
        "engine_tests.execution._distributed_regions",
        "shadowspill.communication.measure",
    )
    torch.compile = forbidden
    triton.runtime.jit.JITFunction._do_compile = forbidden
    triton.runtime.autotuner.Autotuner._bench = forbidden
    reference = torch.load(directory / "reference.pt", weights_only=True)
    dist.init_process_group("gloo", timeout=timedelta(seconds=75))
    assert not torch.cuda.is_initialized()
    logical = 1 - dist.get_rank()
    phase = "resume" if resume else "execute"
    executor = None
    physical = []
    try:
        executor = Executor(
            selected,
            spill=pinned_host(capacity=selected.memory_budgets.spill_bytes),
            device=0,
            rank_map={0: 1, 1: 0},
            reject_overbudget=True,
        )
        checkpoint = directory / f"checkpoint-rank-{logical}.ssp"

        def fill(request, chunk):
            values = reference["initial"][request.name].reshape(-1)
            chunk.copy_(values[request.offset : request.offset + chunk.numel()])

        if resume:
            assert load_checkpoint(executor, checkpoint, chunk_bytes=4096) == {
                "completed": 1
            }
            before = torch.load(
                directory / f"execute-rank-{logical}-state-0.pt", weights_only=True
            )
            count = same_pieces(export(executor), before["tensors"])
            write(
                directory,
                f"resume-checkpoint-rank-{logical}.json",
                dict(status="BITWISE_ALL_OWNED_STATE", elements=count),
            )
        else:
            executor.initialize(fill, chunk_bytes=4096)
        assert adapter_statistics().physical_budget_sealed
        for step in range(1 if resume else 0, 3):
            calls = reference["examples"][step][logical]
            pending = executor.run(
                input_loader=lambda i, calls=calls: calls[i],
                runtime_trace=True,
                accumulate_gradients=reference["expected"][step]["accumulate"],
            )
            result = pending.result()
            assert pending.diagnostics is not None
            trace = pending.diagnostics.as_dict()
            write(directory, f"{phase}-rank-{logical}-trace-{step}.json", trace)
            actual = export(executor)
            torch.save(
                dict(
                    tensors=actual,
                    logical_rank=logical,
                    step=step,
                    losses=[v.detach().cpu().clone() for v in result.losses],
                ),
                directory / f"{phase}-rank-{logical}-state-{step}.pt",
            )
            if step == 0:
                save_checkpoint(
                    executor, checkpoint, metadata={"completed": 1}, chunk_bytes=4096
                )
                executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
                assert load_checkpoint(executor, checkpoint, chunk_bytes=4096) == {
                    "completed": 1
                }
                count = same_pieces(export(executor), actual)
                write(
                    directory,
                    f"checkpoint-rank-{logical}.json",
                    dict(status="BITWISE_ALL_OWNED_STATE", elements=count),
                )
            executor.synchronize()
            assert check_physical_budget() == 0
            physical.append(statistics_dict(adapter_statistics()))
            assert all(
                v.object_id is None
                for v in live_allocations(executor.runtime, "execution")
            )
            print(phase, logical, step, "completed", flush=True)
            del pending, result, actual, trace
        write(directory, f"{phase}-rank-{logical}-physical.json", physical)
        executor.close()
        executor = None
    finally:
        if executor is not None:
            executor.close()
        dist.destroy_process_group()


def inspect(directory, case):
    from dataclasses import asdict

    import torch

    from engine_tests.execution.narrow_regions.state import merge, same_pieces
    from qualification.numerical.metrics import tensor_metrics
    from qualification.numerical.tolerances import (
        LOSS_ABSOLUTE_TOLERANCE,
        LOSS_RELATIVE_TOLERANCE,
        meets_tensor_tolerance,
    )
    from shadowspill import Plan

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    reference = torch.load(directory / "reference.pt", weights_only=True)
    assert reference["reference_contract"] == "ordinary_normal_compiled_cuda"
    selected = Plan.load(directory / "plan.ssp")
    assert not selected.simulation_only_reasons
    assert (selected._accumulation_plan is not None) == (
        case["kind"] == "forward_backward"
    )
    comparisons, failures, signatures = [], [], {}
    for phase, steps in (("execute", range(3)), ("resume", range(1, 3))):
        for step in steps:
            wanted = reference["expected"][step]
            ranks = [
                torch.load(
                    directory / f"{phase}-rank-{rank}-state-{step}.pt",
                    weights_only=True,
                )
                for rank in range(2)
            ]
            actual = merge(ranks, wanted["state"])
            for name, expected in wanted["state"].items():
                value = actual[name]
                category = (
                    "gradient"
                    if name.startswith("gradient.")
                    else "optimizer"
                    if name.startswith("optimizer.")
                    else "model"
                )
                metric = tensor_metrics(expected, value)
                passed = meets_tensor_tolerance(
                    metric,
                    key=f"state/{category}/{name}",
                    step_size=0.001,
                    steps=step + 1,
                )
                if (
                    name.endswith(".step")
                    or name in reference["frozen"]
                    or (category == "model" and case["kind"] == "forward_backward")
                ):
                    passed = torch.equal(value, expected)
                row = dict(
                    phase=phase, step=step, name=name, passed=passed, **asdict(metric)
                )
                comparisons.append(row)
                if not passed:
                    failures.append(row)
            for rank in range(2):
                assert ranks[rank]["logical_rank"] == rank
                for index, (value, expected) in enumerate(
                    zip(ranks[rank]["losses"], wanted["losses"][rank], strict=True)
                ):
                    if not torch.allclose(
                        value,
                        expected,
                        atol=LOSS_ABSOLUTE_TOLERANCE,
                        rtol=LOSS_RELATIVE_TOLERANCE,
                    ):
                        failures.append(
                            dict(
                                phase=phase,
                                step=step,
                                rank=rank,
                                microbatch=index,
                                kind="loss",
                                actual=float(value),
                                expected=float(expected),
                            )
                        )
                if phase == "resume":
                    original = torch.load(
                        directory / f"execute-rank-{rank}-state-{step}.pt",
                        weights_only=True,
                    )
                    same_pieces(original["tensors"], ranks[rank]["tensors"])
                    assert all(
                        torch.equal(a, b)
                        for a, b in zip(
                            original["losses"], ranks[rank]["losses"], strict=True
                        )
                    )
                trace = json.loads(
                    (directory / f"{phase}-rank-{rank}-trace-{step}.json").read_text()
                )
                assert (
                    trace["summary"]["trace_complete"]
                    and not trace["allocator"]["overflow"]
                )
                for key in (
                    "event_overflow",
                    "allocation_event_overflow",
                    "callback_failures_after",
                    "queued_actions_after",
                    "pending_retirements_after",
                ):
                    assert not trace["runtime"][key], (phase, step, rank, key)
                collective = trace["collectives"]
                assert collective["enabled"] and collective["complete"]
                for records in (collective, collective["host_copies"]):
                    assert all(
                        records[key] == 0
                        for key in ("dropped", "errors", "missing_timings")
                    )
                fields = (
                    "operation",
                    "group_ranks",
                    "group_sequence",
                    "input_bytes",
                    "output_bytes",
                    "source",
                    "task_id",
                    "execution_task_id",
                )
                signatures[phase, step, rank] = [
                    [record[field] for field in fields]
                    for record in collective["records"]
                ]
            assert (
                signatures[phase, step, 0]
                and signatures[phase, step, 0] == signatures[phase, step, 1]
            )
        for rank in range(2):
            physical = json.loads(
                (directory / f"{phase}-rank-{rank}-physical.json").read_text()
            )
            assert len(physical) == len(steps)
            assert all(v["physical_budget_sealed"] for v in physical)
    checkpoint = {
        f"{prefix}rank-{rank}": json.loads(
            (directory / f"{prefix}rank-{rank}.json").read_text()
        )
        for prefix in ("checkpoint-", "resume-checkpoint-")
        for rank in range(2)
    }
    assert all(v["status"] == "BITWISE_ALL_OWNED_STATE" for v in checkpoint.values())
    assert not torch.cuda.is_initialized()
    write(
        directory,
        "numerical-inspection.json",
        dict(
            status="FAIL" if failures else "PASS",
            case=case,
            comparisons=comparisons,
            failures=failures,
            checkpoint=checkpoint,
            state_comparisons=len(comparisons),
            collective_invocations=5,
            physical_observations=10,
            fresh_resume="BITWISE",
            thresholds="Unchanged qualification.numerical.tolerances",
            scope=(
                "Actual FP16 parameters/gradients, FP32 masters/Adam state, "
                "generic tensor operations under two-rank DP. "
                "No attention or expert provider qualification."
            ),
        ),
    )
    assert not failures, failures


if __name__ == "__main__":
    directory, phase = Path(sys.argv[1]), sys.argv[2]
    if phase in ("communicate", "execute", "resume"):
        rank = int(os.environ["LOCAL_RANK"])
        devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            devices.split(",")[rank] if devices is not None else str(rank)
        )
    case = json.loads((directory / "case.json").read_text())
    if phase == "resume":
        execute(directory, case, resume=True)
    else:
        globals()[phase](directory, case)
