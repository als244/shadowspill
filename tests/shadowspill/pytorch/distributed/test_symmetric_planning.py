from __future__ import annotations

import json
import tempfile
import traceback
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
from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.planner.admission import (
    AdmissionFacts,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed._control import PreparationError
from shadowspill.pytorch.distributed._plan_exchange import pack
from shadowspill.pytorch.distributed._search import DistributedPlanner
from shadowspill.pytorch.distributed._symmetry import Symmetry, verify
from shadowspill.search.planner import _Planner
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
    selected = _Planner(None, None, tmp_path, None, "contribute", False).plan(
        value, 288, 1024
    )
    received = contract(value).receive(pack(selected, compact_facts=compact_facts))
    assert received.facts == selected.effective_facts
    assert received.admission.simulation == selected.simulation


@pytest.mark.parametrize("difference", ["workspace", "alignment", "pool", "capacity"])
def test_compact_received_plan_rejects_changed_allocation_facts(tmp_path, difference):
    value = problem()
    selected = _Planner(None, None, tmp_path, None, "contribute", False).plan(
        value, 288, 1024
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
    selected = _Planner(None, None, tmp_path, None, "contribute", False).plan(
        value, 288, 1024
    )
    compact = pack(selected, compact_facts=True)
    assert "tasks" not in compact["facts"]
    assert contract(value).receive(compact).facts.tasks is value.admission_facts.tasks
    full = pack(selected)
    assert AdmissionFacts.from_dict(full["facts"]) == selected.effective_facts


def distributed_worker(rank, root, failure):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "rendezvous")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=50),
    )
    bound = Distributed(dist.group.WORLD, symmetric_planning=True, timeout=30)._bind(
        torch.nn.Linear(1, 1), dist.group.WORLD, namespace="symmetric"
    )
    options = SearchOptions(
        workers=1,
        generic=GenericPlanningOptions(
            minimum_object_bytes_evict_eligible=0, deterministic=True
        ),
    )
    planner = DistributedPlanner(
        None, options, Path(root, f"rank-{rank}"), None, "contribute", False, True
    )
    calls = 0
    original_plan = _Planner.plan

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if failure and rank == 0:
            raise RuntimeError("injected CPU search failure")
        return original_plan(self, *args, **kwargs)

    _Planner.plan = counted
    try:
        if failure:
            with (
                pytest.raises(
                    (RuntimeError, PreparationError),
                    match="injected CPU search failure",
                ),
                bound.activate(),
            ):
                planner.prepare_geometry(
                    (problem(rank),), ((288, 1024),), incumbents=True
                )
            return
        with bound.activate():
            local = problem(rank)
            selected = planner.plan(local, 288, 1024)
            assert calls == 1  # two alternatives, exactly one searched per rank
            assert len(selected.result.resolutions) == 2
            assert selected.result.program.devices == local.program.devices
            assert [
                p.compatibility_digest for p in selected.result.program.profiles
            ] == [p.compatibility_digest for p in local.program.profiles]
            assert [p.runtime_ns for p in selected.result.program.profiles] == [
                0,
                30,
                50,
            ]
            assert (
                selected.transfer_bandwidths.fetch_solo_bytes_per_second == 500_000_000
            )
            assert (
                selected.fixed_layout.required_bytes
                <= selected.effective_facts.pool_capacity_bytes
            )
            restored = AnnotatedProgramPlan.from_json(selected.to_json())
            assert restored.simulation.makespan_ns == selected.simulation.makespan_ns
            bound.control.agree(
                "test/selection",
                [item.to_dict() for item in selected.result.selections],
            )

            # Different capacities reject sharing before the candidate search.
            # Local shard groups have independent collective calibrations;
            # the shared plan must use the slowest matching phase everywhere.
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
                program=replace(
                    local.program,
                    alias_groups=(transferred,),
                ),
            )
            shared, evidence = verify(phased)
            assert shared is not None
            assert shared.program.alias_groups[0].transfer.fetch[1].duration_ns == 20
            assert evidence["mode"] == "symmetric"

            # Autotuned kernels can use different temporary sizes without
            # changing outputs, allocation order, or the task's peak workspace.
            task = local.admission_facts.tasks[-1]
            varied = replace(
                local,
                program=replace(
                    local.program,
                    profiles=tuple(
                        replace(profile, runtime_ns=500)
                        if profile.profile_id == "save"
                        else profile
                        for profile in local.program.profiles
                    ),
                ),
                admission_facts=replace(
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
            shared, evidence = verify(contract(varied))
            assert shared is not None
            assert shared.facts.tasks[-1].allocation_steps[0].charged_bytes == 2
            assert evidence["allocation_policy"].startswith("max_anonymous")
            varied_plan = planner.plan(varied, 288, 1024)
            assert varied_plan.result.selections[0].option_id == "recompute"
            assert (
                varied_plan.effective_facts.tasks[-1].allocation_steps[0].charged_bytes
                == 2
            )
            planner.prepare_geometry((varied,), ((288, 1024),), incumbents=True)
            shared_answer = planner.answer(varied, 288, 1024, None)
            assert shared_answer.plan is not None
            assert (
                shared_answer.plan.effective_facts.tasks[-1]
                .allocation_steps[0]
                .charged_bytes
                == 2
            )

            # Different capacities reject sharing before the candidate search.
            unequal = replace(contract(local), scratch=rank)
            shared, evidence = verify(unequal)
            assert shared is None and "scratch_reserve" in evidence["reason"]

            unequal_problem = replace(local, dynamic_scratch_reserve_bytes=rank)
            calls = 0
            planner.prepare_geometry(
                (unequal_problem,), ((288, 1024),), incumbents=True
            )
            assert not planner._prepared_answers
            fallback = planner.answer(unequal_problem, 288, 1024, None)
            assert fallback.plan is not None and calls == 2

            second = replace(
                local,
                program=replace(
                    local.program,
                    profiles=tuple(
                        replace(p, runtime_ns=p.runtime_ns * 2)
                        for p in local.program.profiles
                    ),
                ),
            )
            calls = 0
            earlier = set(Path(root, f"rank-{rank}").rglob("distributed/plans/*.json"))
            budgets = ((288, 1024), (352, 1024))
            planner.prepare_geometry((local, second, local), budgets, incumbents=True)
            assert calls == 2  # one ordering, both budgets, on each rank
            for value in (local, second, local):
                prior = None
                for budget in budgets:
                    answer = planner.answer(value, *budget, None)
                    assert answer.plan is not None
                    assert len(answer.plan.result.resolutions) == 2
                    if prior is not None:
                        assert answer.makespan_ns <= prior
                    prior = answer.makespan_ns
                    bound.control.agree(
                        "test/batched_choice",
                        [item.to_dict() for item in answer.plan.result.selections],
                    )
            assert calls == 2
            saved = (
                set(Path(root, f"rank-{rank}").rglob("distributed/plans/*.json"))
                - earlier
            )
            assert len(saved) == 4
            for path in saved:
                assert len(json.loads(path.read_text())["plan"]["resolutions"]) == 2

            # Resume with warm owner caches. Every rank must still receive all
            # retained resolutions, even if its earlier sidecars were lost.
            for path in saved:
                path.unlink()
            planner.prepare_geometry((local, second, local), budgets, incumbents=True)
            for value in (local, second, local):
                for budget in budgets:
                    answer = planner.answer(value, *budget, None)
                    assert len(answer.plan.result.resolutions) == 2
            rebuilt = (
                set(Path(root, f"rank-{rank}").rglob("distributed/plans/*.json"))
                - earlier
            )
            assert rebuilt
            for path in rebuilt:
                assert len(json.loads(path.read_text())["plan"]["resolutions"]) == 2
    except BaseException:
        Path(root, f"failure-rank-{rank}.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        _Planner.plan = original_plan
        bound.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", [False, True])
def test_shared_search_with_real_cpu_processes(failure):
    with tempfile.TemporaryDirectory() as root:
        try:
            mp.spawn(distributed_worker, args=(root, failure), nprocs=2, join=True)
        finally:
            for path in Path(root).glob("failure-rank-*.txt"):
                print(path.read_text())


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
