"""Build/measure local DP compute; reduce partial gradients inside sharded spills."""

import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from shadowspill import (
    MemoryBudgets,
    Parallelism,
    apply_profiles,
    build_forward_backward,
    measure_program,
    plan_program,
)
from shadowspill.communication.requests import collective_requests
from shadowspill.ir import MemoryLocation
from shadowspill.step.ordering import StepDataOrdering
from tests.shadowspill.build._shard_loading import check_shard_loading


@pytest.mark.cuda
def test_one_gpu_build_profiles_compute_and_plans_sharded_gradient_spills(
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
    check_shard_loading(source)
    assert source.parallelism.world_size == 4
    assert not source.bindings["gradient_templates"]
    assert not source.bindings["collective_tasks"]
    gradient = source.bindings["persistent_gradients"][0]["gradient_object_id"]
    aliases = {item.object_id: item.alias_group_id for item in source.objects}
    shard = source.bindings["gradient_shards"][aliases[gradient]]
    assert (shard["logical_bytes"], shard["padded_bytes"], shard["shard_bytes"]) == (
        140,
        144,
        72,
    )
    assert tuple(map(tuple, shard["groups"])) == ((0, 1), (2, 3))
    assert tuple(map(tuple, shard["replica_groups"])) == ((0, 2), (1, 3))
    profiles = measure_program(source, warmup_seconds=0.001, measure_seconds=0.002)
    # Publication adopts the actual compiled allocation, not merely its tensor
    # shape. Equal-shard padding must already exist in every first backward.
    for form in source.bindings["training_templates"][0][0]:
        if form["accumulates"]:
            continue
        backward = form["backward"]
        leaf = next(
            slot["leaf_index"]
            for slot in backward["contributions"]
            if slot["object_id"] == gradient
        )
        allocation = next(
            event
            for event in profiles.tasks[backward["profile_key"]].metadata[
                "allocation_trace"
            ]
            if leaf in event["output_leaf_indices"]
        )
        assert allocation["requested_bytes"] == shard["padded_bytes"]
    # Compute profiles do not need distributed memory or timing measurements.
    profiled = apply_profiles(source, profiles)
    from tests.shadowspill.offline._communication import estimated_communication

    info = estimated_communication(
        source,
        topology={"node0": (0, 1, 2, 3)},
    )
    budgets = MemoryBudgets(profiled.fixed_device_bytes + (8 << 20), 8 << 20)
    for ordering in (
        StepDataOrdering(depth=2, breadth=1),
        StepDataOrdering(depth=1, breadth=2),
    ):
        plan = plan_program(
            profiled, memory_budgets=budgets, communication_info=info, ordering=ordering
        )
        for path, accumulate in ((plan, False), (plan._accumulation_plan, True)):
            assert path is not None
            initial = {row.alias_group_id for row in path.program.initial_residency}
            assert (aliases[gradient] in initial) == accumulate
            assert not any(
                task.phase == "gradient_communication" for task in path.program.tasks
            )
            assert any(
                row.alias_group_id == aliases[gradient]
                and row.location == MemoryLocation.SPILL
                for row in path.program.final_residency
            )
            transfer = next(
                a.transfer
                for a in path._costed_program.alias_groups
                if a.alias_group_id == aliases[gradient]
            )
            assert transfer.spill_size_bytes == 72
            assert [
                phase.timing_class for phase in transfer.evict if phase.kind == "timed"
            ] == ["reduce_scatter_ring+all_reduce_ring"]
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
from tests.shadowspill.offline._communication import estimated_communication
profiled = ProfiledProgram.load(sys.argv[1])
plan = plan_program(profiled,
    memory_budgets=MemoryBudgets(profiled.fixed_device_bytes + (8 << 20), 8 << 20),
    communication_info=estimated_communication(profiled.program,
        topology={'node0': (0, 1, 2, 3)}))
plan.save(sys.argv[2])
loaded = Plan.load(sys.argv[2])
assert loaded.summary['logical_ranks'] == 4
assert loaded._accumulation_plan is not None
assert loaded.simulation_only_reasons
assert not any(task.phase == 'gradient_communication' for task in loaded.program.tasks)
assert loaded.program.bindings['gradient_shards']
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
