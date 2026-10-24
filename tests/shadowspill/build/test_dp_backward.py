"""Build/measure DP gradients on one GPU, then plan both gradient modes offline."""

import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    Parallelism,
    TaskProfile,
    apply_profiles,
    build_forward_backward,
    measure_program,
    plan_program,
)
from shadowspill.communication.requests import collective_requests
from shadowspill.ir import MemoryLocation
from shadowspill.pytorch.executable import restore_executable
from shadowspill.step.ordering import StepDataOrdering


@pytest.mark.cuda
def test_one_gpu_build_profiles_padding_and_owned_gradient_accumulation(
    monkeypatch, tmp_path
):
    if not torch.cuda.is_available():
        pytest.skip("requires representative CUDA GPU")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "DP computation preparation must not create process groups"
        )

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    model = nn.Linear(7, 5, bias=False, dtype=torch.bfloat16, device="meta")
    source = build_forward_backward(
        model,
        objective=lambda m, x: m(x).float().square().mean(),
        example_args=(torch.empty(2, 7, dtype=torch.bfloat16),),
        num_examples=2,
        partition="whole",
        grad_dtype=torch.float32,
        parallelism=Parallelism.dp(ranks_per_replica=2, num_replicas=2),
    )
    source.save(tmp_path / "program.ssp")
    assert source.parallelism.world_size == 4
    assert len(source.bindings["gradient_templates"]) == 1
    completion = source.bindings["gradient_templates"][0]
    assert completion["padding"] is not None
    assert [phase["operation"] for phase in completion["phases"]] == [
        "reduce_scatter",
        "all_reduce",
    ]
    padding = restore_executable(
        source.executables[completion["padding"]["profile_key"]]
    )
    gradient = torch.arange(35, dtype=torch.float32, device="cuda").reshape(5, 7)
    (padded,) = padding(gradient)
    torch.testing.assert_close(padded[:35], gradient.reshape(-1))
    assert padded.shape == (36,) and padded[-1].item() == 0
    addition = restore_executable(source.executables[completion["accumulation_key"]])
    old = torch.full((18,), 7.0, device="cuda")
    contribution = padded[:18].clone()
    expected = old + contribution
    addition(old, contribution)
    torch.testing.assert_close(old, expected)
    addition(old, contribution)
    torch.testing.assert_close(old, expected + contribution)
    profiles = measure_program(source, warmup_seconds=0.001, measure_seconds=0.002)
    assert completion["padding"]["profile_key"] in profiles.tasks
    assert completion["accumulation_key"] in profiles.tasks
    # Collective scratch remains explicitly unresolved until supplied. These
    # assumptions are for CPU simulation, not distributed execution admission.
    with pytest.raises(ValueError, match="workspace_bytes"):
        apply_profiles(source, profiles)
    for key in source.bindings["collective_tasks"]:
        profiles.tasks[key] = TaskProfile(workspace_bytes=0)
    profiled = apply_profiles(source, profiles)
    info = CommunicationInfo(
        20e9,
        20e9,
        topology={"node0": (0, 1, 2, 3)},
        scale_up_bytes_per_second=100e9,
        scale_up_latency_seconds=2e-6,
    )
    budgets = MemoryBudgets(profiled.fixed_device_bytes + (8 << 20), 8 << 20)
    for ordering in (
        StepDataOrdering(depth=2, breadth=1),
        StepDataOrdering(depth=1, breadth=2),
    ):
        plan = plan_program(
            profiled, memory_budgets=budgets, communication_info=info, ordering=ordering
        )
        stored, local, new = (
            completion["stored"],
            completion["source"],
            completion["new"],
        )
        aliases = {item.object_id: item.alias_group_id for item in source.objects}
        for path, accumulate in ((plan, False), (plan._accumulation_plan, True)):
            assert path is not None
            initial = {row.alias_group_id for row in path.program.initial_residency}
            assert (aliases[stored] in initial) == accumulate
            assert aliases[local] not in initial
            reductions = [
                task
                for task in path.program.tasks
                if task.phase == "gradient_communication"
            ]
            if accumulate:
                assert all(stored not in task.inputs for task in reductions)
            scatter = next(task for task in reductions if task.outputs)
            assert scatter.outputs == ((new if accumulate else stored),)
            assert any(
                row.alias_group_id == aliases[stored]
                and row.location == MemoryLocation.SPILL
                for row in path.program.final_residency
            )
            assert path.simulation_only_reasons
    requests = collective_requests(source, None)
    assert {row.operation for row in requests} == {
        "all_gather",
        "reduce_scatter",
        "all_reduce",
    }
    profiled.save(tmp_path / "profiled.ssp")
    assert all(parameter.is_meta for parameter in model.parameters())
    code = """
import importlib.abc, sys
class CPUOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if (name.split('.')[0] in ('torch', 'mlops')
            or name.startswith(('shadowspill.runtime', 'shadowspill.build'))):
            raise AssertionError('CPU planning imported ' + name)
sys.meta_path.insert(0, CPUOnly())
from shadowspill import (ProfiledProgram, Plan, CommunicationInfo, MemoryBudgets,
                        plan_program, write_report)
profiled = ProfiledProgram.load(sys.argv[1])
plan = plan_program(profiled,
    memory_budgets=MemoryBudgets(profiled.fixed_device_bytes + (8 << 20), 8 << 20),
    communication_info=CommunicationInfo(20e9, 20e9,
        topology={'node0': (0, 1, 2, 3)}, scale_up_bytes_per_second=100e9,
        scale_up_latency_seconds=2e-6))
plan.save(sys.argv[2])
loaded = Plan.load(sys.argv[2])
assert loaded.summary['logical_ranks'] == 4
assert loaded._accumulation_plan is not None
assert loaded.simulation_only_reasons
assert any(task.phase == 'gradient_communication' for task in loaded.program.tasks)
write_report(loaded, sys.argv[3])
print('CPU-only reduced-gradient planning and archive roundtrip passed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            code,
            str(tmp_path / "profiled.ssp"),
            str(tmp_path / "plan.ssp"),
            str(tmp_path / "report"),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "roundtrip passed" in result.stdout
