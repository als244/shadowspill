from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    MutationSpec,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    SharedResidencyPolicy,
    TaskAlternativeGroup,
    TaskAlternativeOption,
    TaskProfile,
    TaskSpec,
    TransferPhase,
    TransferSpec,
)
from shadowspill.planner import plan_program
from shadowspill.planner.admission import (
    AdmissionFacts,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed._plan_exchange import pack
from shadowspill.pytorch.distributed._symmetry import Symmetry, verify
from shadowspill.simulator import SimulationConfig


def problem(rank=0):
    device = f"cuda_{rank}"
    program = ShadowSpillProgram(
        devices=(DeviceSpec(device, f"process_{rank}", "cuda", rank),),
        alias_groups=(AliasGroupSpec("state", device, 64, retain_spill_copy=True),),
        objects=(ObjectSpec("state_object", "state", 0, 64),),
        profiles=(
            TaskProfile("begin", 0, 0, "control"),
            TaskProfile("save", 20 + 10 * rank, 0, f"save_abi_{rank}"),
            TaskProfile("recompute", 40 + 10 * rank, 8, f"recompute_abi_{rank}"),
        ),
        tasks=(
            TaskSpec(
                "begin",
                ResourceSpec(device, ResourceKind.CONTROL),
                "begin",
                requires_entrypoint=False,
            ),
            TaskSpec(
                "save",
                ResourceSpec(device, ResourceKind.COMPUTE),
                "save",
                dependencies=("begin",),
                inputs=("state_object",),
            ),
            TaskSpec(
                "recompute",
                ResourceSpec(device, ResourceKind.COMPUTE),
                "recompute",
                dependencies=("begin",),
                inputs=("state_object",),
            ),
        ),
        task_alternative_groups=(
            TaskAlternativeGroup(
                "choice",
                (
                    TaskAlternativeOption("save", ("save",), ("state",)),
                    TaskAlternativeOption("recompute", ("recompute",)),
                ),
            ),
        ),
    )
    facts = AdmissionFacts(
        device,
        256,
        192,
        8,
        (
            TaskAdmissionSpec("begin"),
            TaskAdmissionSpec("save"),
            TaskAdmissionSpec(
                "recompute",
                workspace_extents=(8,),
                allocation_steps=(
                    TaskAllocationStep(0, TaskAllocationStepKind.ALLOCATE, 8),
                    TaskAllocationStep(0, TaskAllocationStepKind.RELEASE),
                ),
            ),
        ),
    )
    return ShadowSpillPlanningProblem(
        role="step",
        program=program,
        initial_residency=(ResidencySpec("state", MemoryLocation.SPILL),),
        final_residency=(ResidencySpec("state", MemoryLocation.SPILL),),
        simulation_config=SimulationConfig.single_device(
            device,
            device_capacity_bytes=192,
            spill_capacity_bytes=1024,
            fetch_solo_bandwidth_bytes_per_second=1_000_000_000 // (rank + 1),
            fetch_concurrent_bandwidth_bytes_per_second=(1_000_000_000 // (rank + 1)),
            evict_solo_bandwidth_bytes_per_second=1_000_000_000,
            evict_concurrent_bandwidth_bytes_per_second=(1_000_000_000),
        ),
        admission_facts=facts,
        source_execution_budget_bytes=288,
        maximum_execution_budget_bytes=512,
        maximum_spill_budget_bytes=2048,
        fixed_execution_bytes=32,
        object_reserve_bytes=64,
        dynamic_scratch_reserve_bytes=0,
    )


def contract(value):
    return Symmetry(
        value.program,
        value.simulation_config,
        value.admission_facts,
        value.initial_residency,
        value.final_residency,
        0,
    )


@pytest.mark.parametrize(
    "difference",
    ["workspace", "alias", "mutation", "allocation", "budget", "residency", "scratch"],
)
def test_symmetry_rejects_memory_or_state_contract_differences(difference):
    value = contract(problem())
    if difference == "workspace":
        value = replace(
            value,
            program=replace(
                value.program,
                profiles=tuple(
                    replace(p, workspace_bytes=p.workspace_bytes + 1)
                    for p in value.program.profiles
                ),
            ),
        )
    elif difference == "alias":
        value = replace(
            value,
            program=replace(
                value.program,
                alias_groups=(replace(value.program.alias_groups[0], size_bytes=80),),
            ),
        )
    elif difference == "mutation":
        value = replace(
            value,
            program=replace(
                value.program,
                tasks=tuple(
                    replace(t, mutations=(MutationSpec("state_object"),))
                    if t.task_id == "save"
                    else t
                    for t in value.program.tasks
                ),
            ),
        )
    elif difference == "allocation":
        value = replace(value, facts=replace(value.facts, minimum_alignment=16))
    elif difference == "budget":
        value = replace(value, facts=replace(value.facts, pool_capacity_bytes=255))
    elif difference == "residency":
        value = replace(value, initial=(ResidencySpec("state", MemoryLocation.DEVICE),))
    else:
        value = replace(value, scratch=1)
    assert value.requirements(None, None) != contract(problem()).requirements(
        None, None
    )


def test_symmetry_preserves_rank_local_abi_and_device_binding():
    assert contract(problem(0)).requirements(None, None) == contract(
        problem(1)
    ).requirements(None, None)


@pytest.mark.parametrize("compact_facts", [False, True])
def test_received_plan_uses_normal_admission_for_shared_residency(
    tmp_path, compact_facts
):
    value = problem()
    group = value.program.task_alternative_groups[0]
    value = replace(
        value,
        program=replace(
            value.program,
            alias_groups=(
                replace(
                    value.program.alias_groups[0],
                    shared_residency=SharedResidencyPolicy.SHARED_READ_ONLY,
                ),
            ),
            task_alternative_groups=(
                replace(
                    group,
                    options=tuple(
                        replace(option, retained_alias_group_ids=())
                        for option in group.options
                    ),
                ),
            ),
        ),
        initial_residency=(),
        final_residency=(),
        admission_facts=replace(
            value.admission_facts, pool_capacity_bytes=192, object_capacity_bytes=128
        ),
    )
    selected = plan_program(
        value,
        execution_budget=288,
        spill_budget=1024,
        artifact_store=tmp_path,
        verbose=False,
    )
    received = contract(value).receive(pack(selected, compact_facts=compact_facts))
    assert received.facts == selected.effective_facts
    assert received.admission.simulation == selected.simulation


@pytest.mark.parametrize("difference", ["workspace", "alignment", "pool", "capacity"])
def test_compact_received_plan_rejects_changed_allocation_facts(tmp_path, difference):
    value = problem()
    selected = plan_program(
        value,
        execution_budget=288,
        spill_budget=1024,
        artifact_store=tmp_path,
        verbose=False,
    )
    payload = pack(selected, compact_facts=True)
    local = contract(value)
    if difference == "capacity":
        payload["facts"]["object_capacity_bytes"] -= 8
    else:
        facts = local.facts
        if difference == "workspace":
            facts = replace(
                facts,
                tasks=(
                    *facts.tasks[:-1],
                    replace(facts.tasks[-1], workspace_extents=(16,)),
                ),
            )
        elif difference == "alignment":
            facts = replace(facts, minimum_alignment=16)
        else:
            facts = replace(facts, pool_capacity_bytes=264)
        local = replace(local, facts=facts)
    with pytest.raises(ValueError, match="verified allocation contract"):
        local.receive(payload)


def test_compact_plan_keeps_persisted_facts_self_contained(tmp_path):
    value = problem()
    selected = plan_program(
        value,
        execution_budget=288,
        spill_budget=1024,
        artifact_store=tmp_path,
        verbose=False,
    )
    compact = pack(selected, compact_facts=True)
    assert "tasks" not in compact["facts"]
    assert contract(value).receive(compact).facts.tasks is value.admission_facts.tasks
    full = pack(selected)
    assert AdmissionFacts.from_dict(full["facts"]) == selected.effective_facts


def distributed_worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "rendezvous")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    bound = Distributed(dist.group.WORLD, symmetric_planning=True, timeout=20)._bind(
        torch.nn.Linear(1, 1), dist.group.WORLD, namespace="symmetric"
    )
    try:
        with bound.activate():
            local = problem(rank)
            shared, evidence = verify(contract(local))
            assert shared is not None and evidence["mode"] == "symmetric"
            assert [p.runtime_ns for p in shared.program.profiles] == [0, 30, 50]
            assert (
                shared.config.devices[0].fetch_solo_bandwidth_bytes_per_second
                == 500_000_000
            )
            transferred = replace(
                local.program.alias_groups[0],
                transfer=TransferSpec(
                    32,
                    (
                        TransferPhase("copy", bytes=32),
                        TransferPhase(
                            "timed",
                            duration_ns=10 + rank * 10,
                            resource="sharded-collectives",
                            submission_ordered=True,
                        ),
                    ),
                    (TransferPhase("copy", bytes=32),),
                ),
            )
            phased = replace(
                contract(local),
                program=replace(local.program, alias_groups=(transferred,)),
            )
            shared, _ = verify(phased)
            assert shared is not None
            assert shared.program.alias_groups[0].transfer.fetch[1].duration_ns == 20
            task = local.admission_facts.tasks[-1]
            varied = replace(
                contract(local),
                facts=replace(
                    local.admission_facts,
                    tasks=(
                        *local.admission_facts.tasks[:-1],
                        replace(
                            task,
                            allocation_steps=(
                                TaskAllocationStep(
                                    1, TaskAllocationStepKind.ALLOCATE, rank + 1
                                ),
                                TaskAllocationStep(1, TaskAllocationStepKind.RELEASE),
                                *task.allocation_steps,
                            ),
                        ),
                    ),
                ),
            )
            shared, evidence = verify(varied)
            assert shared is not None
            assert shared.facts.tasks[-1].allocation_steps[0].charged_bytes == 2
            assert evidence["allocation_policy"].startswith("max_anonymous")
            shared, evidence = verify(replace(contract(local), scratch=rank))
            assert shared is None and "scratch_reserve" in evidence["reason"]
    finally:
        bound.close()
        dist.destroy_process_group()


def test_rank_profiles_reconcile_with_real_cpu_processes():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(distributed_worker, args=(root,), nprocs=2, join=True)


def test_reconciliation_rejects_combined_workspace_peak():
    from shadowspill.pytorch.distributed._allocation_symmetry import reconcile

    task = TaskAdmissionSpec(
        "task",
        workspace_extents=(2, 6),
        allocation_steps=(
            TaskAllocationStep(0, TaskAllocationStepKind.ALLOCATE, 6),
            TaskAllocationStep(1, TaskAllocationStepKind.ALLOCATE, 2),
            TaskAllocationStep(0, TaskAllocationStepKind.RELEASE),
            TaskAllocationStep(1, TaskAllocationStepKind.RELEASE),
        ),
    )
    facts = AdmissionFacts("cuda_0", 16, 16, 1, (task,))

    class Peers:
        def exchange(self, label, value):
            if label.endswith("allocation_structure"):
                return [value, value]
            return [value, {"vectors": [(2, 6)], "indices": [0]}]

    assert reconcile(facts, Peers()) is None
