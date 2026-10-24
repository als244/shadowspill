"""Shard-local optimizer bindings and explicit completion/publication tasks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    Parallelism,
    TaskProfile,
    apply_profiles,
    build_step,
    measure_program,
    plan_program,
)
from shadowspill.artifacts.tensors import tensor_geometry
from shadowspill.parallelism.views import owned_view_offsets
from shadowspill.pytorch.executable import restore_executable


@pytest.mark.cuda
@pytest.mark.parametrize("master_dtype", [None, torch.float32])
def test_step_build_measure_plan_and_local_owned_updates(
    monkeypatch, master_dtype, tmp_path
):
    if not torch.cuda.is_available():
        pytest.skip("requires representative CUDA GPU")

    def forbidden(*args, **kwargs):
        raise AssertionError("DP task preparation must not create process groups")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    model = nn.Linear(7, 5, dtype=torch.bfloat16, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    program = build_step(
        model,
        optimizer=optimizer,
        objective=lambda m, x: m(x).float().square().mean(),
        example_args=(torch.empty(2, 7, dtype=torch.bfloat16),),
        num_examples=2,
        partition="whole",
        grad_dtype=torch.float32,
        master_dtype=master_dtype,
        parallelism=Parallelism.dp(ranks_per_replica=2, num_replicas=2),
    )
    assert not optimizer.state
    assert all(parameter.is_meta for parameter in model.parameters())
    assert program.bindings["distributed_training_lowered"] == "step"
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    for key in program.bindings["collective_tasks"]:
        profiles.tasks[key] = TaskProfile(workspace_bytes=0)
    profiled = apply_profiles(program, profiles)
    info = CommunicationInfo(
        20e9,
        20e9,
        topology={"node": (0, 1, 2, 3)},
        scale_up_bytes_per_second=100e9,
        scale_up_latency_seconds=2e-6,
    )
    for ordering in ("tail", "stage_interleaved"):
        plan = plan_program(
            profiled,
            memory_budgets=MemoryBudgets(
                profiled.fixed_device_bytes + (8 << 20), 8 << 20
            ),
            communication_info=info,
            optimizer_ordering=ordering,
        )
        assert plan.simulation_only_reasons
        assert plan._accumulation_plan is None
        _check_completion(plan)
    _check_local_math(program, master_dtype)
    if master_dtype is not None:
        profiled.save(tmp_path / "profiled.ssp")
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("_cpu_plan_step.py")),
                str(tmp_path / "profiled.ssp"),
                str(tmp_path),
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "roundtrip passed" in result.stdout


def _check_completion(plan):
    objects = {item.object_id: item for item in plan.program.objects}
    gradients = {item["stored"] for item in plan.program.bindings["gradient_templates"]}
    tasks = plan.program.tasks
    intervals = {item.task_id: item for item in plan.timeline.task_intervals}
    for update in (task for task in tasks if task.phase == "optimizer"):
        assert gradients.intersection(update.inputs)
        aliases = {objects[value].alias_group_id for value in update.inputs}
        for earlier in tasks:
            if earlier.phase not in {"forward", "backward", "gradient_communication"}:
                continue
            if (
                aliases.intersection(
                    objects[value].alias_group_id
                    for value in (*earlier.inputs, *earlier.outputs)
                )
                and earlier.task_id in intervals
            ):
                assert (
                    intervals[update.task_id].start_ns
                    >= intervals[earlier.task_id].end_ns
                )
    for publication in (task for task in tasks if task.phase == "weight_communication"):
        owned, full = (objects[value] for value in publication.inputs)
        assert owned.alias_group_id == full.alias_group_id
        assert owned.size_bytes * 2 == full.size_bytes
        assert not publication.outputs
        assert tuple(item.object_id for item in publication.mutations) == (
            full.object_id,
        )


def _check_local_math(program, master_dtype):
    """Run actual saved update payloads on four logical owners, one GPU at a time."""
    torch.manual_seed(19)
    geometry = tensor_geometry(program)
    optimizer_bindings = program.bindings["optimizer_bindings"]
    parameter_names = {
        row["name"]: row["object_id"] for row in program.bindings["registrations"]
    }
    views = program.bindings["owned_parameter_views"]
    target_dtype = master_dtype or torch.bfloat16
    reference = {
        name: nn.Parameter(
            torch.randn(
                geometry[value]["shape"], device="cuda", dtype=torch.bfloat16
            ).to(target_dtype)
        )
        for name, value in parameter_names.items()
    }
    teacher = torch.optim.AdamW(reference.values(), lr=0.001, foreach=False)
    ranks = []
    for rank in range(4):
        offsets = owned_view_offsets(program, rank)
        full_weights = {
            name: torch.nn.functional.pad(
                parameter.detach().to(torch.bfloat16).reshape(-1),
                (0, parameter.numel() % 2),
            )
            for name, parameter in reference.items()
        }
        values = {}
        for binding in optimizer_bindings:
            name, role = binding["name"], binding["role"]
            dtype = getattr(torch, binding["dtype"].removeprefix("torch."))
            view = views.get(binding["object_id"])
            if view is not None:
                width = view["elements"]
                values[name] = full_weights[view["parameter_name"]].narrow(
                    0, offsets[binding["object_id"]], width
                )
            elif role == "parameter":
                width = binding["shape"][0]
                values[name] = (
                    full_weights[name]
                    .narrow(0, (rank % 2) * width, width)
                    .to(dtype)
                    .clone()
                )
            else:
                values[name] = torch.zeros(
                    binding["shape"], dtype=dtype, device=binding["device"]
                )
        ranks.append((values, full_weights))
    functions = {
        row["profile_key"]: restore_executable(program.executables[row["profile_key"]])
        for row in program.bindings["optimizer_templates"]
    }
    names_by_object = {
        row["object_id"]: row["name"]
        for row in optimizer_bindings
        if row["object_id"] is not None
    }
    for _ in range(3):
        gradients = {
            name: torch.randn_like(value, dtype=torch.float32)
            for name, value in reference.items()
        }
        for name, value in reference.items():
            value.grad = gradients[name].to(value.dtype)
        teacher.step()
        for rank, (values, _) in enumerate(ranks):
            for name, gradient in gradients.items():
                padded = torch.nn.functional.pad(
                    gradient.reshape(-1), (0, gradient.numel() % 2)
                )
                width = padded.numel() // 2
                values["gradient." + name].copy_(padded[(rank % 2) * width :][:width])
            for component in program.bindings["optimizer_templates"]:
                arguments = {
                    slot["leaf_index"]: values[names_by_object[slot["object_id"]]]
                    for slot in component["inputs"]
                }
                arguments.update(
                    {
                        slot["leaf_index"]: values[slot["name"]]
                        for slot in component["host_inputs"]
                    }
                )
                with torch.no_grad():
                    functions[component["profile_key"]](
                        *(arguments[index] for index in range(len(arguments)))
                    )
        # Emulate publication only; distributed transport is tested separately.
        for name, expected in reference.items():
            compute_name = ("compute." if master_dtype is not None else "") + name
            for replica in range(2):
                owners = ranks[replica * 2 : replica * 2 + 2]
                master = torch.cat([values[name] for values, _ in owners])
                torch.testing.assert_close(
                    master[: expected.numel()], expected.reshape(-1)
                )
                result = torch.cat([values[compute_name] for values, _ in owners])
                for _, weights in owners:
                    weights[name].copy_(result)
                torch.testing.assert_close(
                    result[: expected.numel()], expected.to(torch.bfloat16).reshape(-1)
                )
