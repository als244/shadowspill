from __future__ import annotations

from dataclasses import replace

from reference.python.admission import replay_admission
from shadowspill.ir import (
    AliasGroupSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    MutationSpec,
    ObjectSpec,
    Program,
    ResidencySpec,
    SharedResidencyPolicy,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionTopology,
    StorageHandoff,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.pytorch.planning.admission.admission_replay import (
    AdmissionReplayPurpose,
    OwnershipTransitionKind,
)
from shadowspill.runtime import AdmissionReplayOperationKind
from tests.shadowspill.planner._examples import COMPUTE, DEVICE


def _program(
    alias_groups: tuple[AliasGroupSpec, ...],
    objects: tuple[ObjectSpec, ...],
    tasks: tuple[TaskSpec, ...],
    *,
    workspace_bytes: int = 0,
    runtime_ns: int = 10,
) -> Program:
    return Program(
        devices=(DEVICE,),
        alias_groups=alias_groups,
        objects=objects,
        profiles=(TaskProfile("profile", runtime_ns, workspace_bytes, "abi"),),
        tasks=tasks,
    )


def _task_admission(
    task_id: str,
    *,
    workspace_extents: tuple[int, ...] = (),
    fresh_outputs: tuple[tuple[str, int], ...] = (),
    replacements: tuple[tuple[str, int], ...] = (),
) -> TaskAdmissionSpec:
    """Build explicit allocation evidence for a hand-authored replay fixture."""

    steps: list[TaskAllocationStep] = []
    workspace_ordinals: list[int] = []
    for extent in workspace_extents:
        ordinal = len(steps)
        workspace_ordinals.append(ordinal)
        steps.append(
            TaskAllocationStep(
                ordinal,
                TaskAllocationStepKind.ALLOCATE,
                extent,
            )
        )
    for alias_id, extent in (*fresh_outputs, *replacements):
        steps.append(
            TaskAllocationStep(
                len(steps),
                TaskAllocationStepKind.ALLOCATE,
                extent,
                alias_id,
            )
        )
    steps.extend(
        TaskAllocationStep(ordinal, TaskAllocationStepKind.RELEASE)
        for ordinal in workspace_ordinals
    )
    return TaskAdmissionSpec(
        task_id,
        workspace_extents=workspace_extents,
        fresh_output_aliases=tuple(alias for alias, _ in fresh_outputs),
        replacement_aliases=tuple(alias for alias, _ in replacements),
        allocation_steps=tuple(steps),
    )


def _empty_topology(program: Program, pool_bytes: int) -> AdmissionTopology:
    return AdmissionTopology(
        "cuda_0",
        pool_bytes,
        pool_bytes,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )


def test_admission_replay_reserves_workspace_for_complete_task_interval() -> None:
    program = _program(
        (
            AliasGroupSpec("input", "cuda_0", 64),
            AliasGroupSpec("output", "cuda_0", 32),
        ),
        (
            ObjectSpec("input_object", "input", 0, 64),
            ObjectSpec("output_object", "output", 0, 32),
        ),
        (
            TaskSpec(
                "forward",
                COMPUTE,
                "profile",
                inputs=("input_object",),
                outputs=("output_object",),
            ),
        ),
        workspace_bytes=32,
    )
    schedule = MemorySchedule(
        (ResidencySpec("input", MemoryLocation.DEVICE),),
        (),
        (ResidencySpec("output", MemoryLocation.DEVICE),),
    )

    replay = replay_admission(
        program,
        schedule,
        topology=AdmissionTopology(
            "cuda_0",
            128,
            128,
            1,
            (
                _task_admission(
                    "forward",
                    workspace_extents=(32,),
                    fresh_outputs=(("output", 32),),
                ),
            ),
        ),
    )

    assert replay.pool.peak_allocated_bytes == 128
    assert replay.pool.final_allocated_bytes == 96
    assert replay.workspace_bytes_by_task == (("forward", 32),)
    purposes = tuple(item.purpose for item in replay.operations)
    assert AdmissionReplayPurpose.TASK_WORKSPACE in purposes
    assert AdmissionReplayPurpose.TASK_OUTPUT in purposes


def test_admission_replay_treats_shared_input_as_externally_resident() -> None:
    program = _program(
        (
            AliasGroupSpec(
                "shared",
                "cuda_0",
                64,
                shared_residency=SharedResidencyPolicy.SHARED_READ_ONLY,
            ),
            AliasGroupSpec("output", "cuda_0", 32),
        ),
        (
            ObjectSpec("shared_object", "shared", 0, 64),
            ObjectSpec("output_object", "output", 0, 32),
        ),
        (
            TaskSpec(
                "forward",
                COMPUTE,
                "profile",
                inputs=("shared_object",),
                outputs=("output_object",),
            ),
        ),
    )
    schedule = MemorySchedule(
        (),
        (),
        (ResidencySpec("output", MemoryLocation.DEVICE),),
    )
    topology = AdmissionTopology(
        "cuda_0",
        32,
        32,
        1,
        (_task_admission("forward", fresh_outputs=(("output", 32),)),),
    )

    replay = replay_admission(
        program,
        schedule,
        topology=topology,
    )

    assert replay.pool.peak_allocated_bytes == 32
    assert all(step.alias_group_id != "shared" for step in replay.operations)


def test_task_local_reuse_preserves_one_physical_lease() -> None:
    program = _program(
        (),
        (),
        (TaskSpec("task", COMPUTE, "profile"),),
        workspace_bytes=32,
    )
    schedule = MemorySchedule((), (), ())
    topology = AdmissionTopology(
        "cuda_0",
        32,
        32,
        1,
        (
            TaskAdmissionSpec(
                "task",
                workspace_extents=(32,),
                allocation_steps=(
                    TaskAllocationStep(
                        0,
                        TaskAllocationStepKind.ALLOCATE,
                        32,
                    ),
                    TaskAllocationStep(0, TaskAllocationStepKind.RELEASE),
                    TaskAllocationStep(
                        1,
                        TaskAllocationStepKind.ALLOCATE,
                        32,
                        reuses_allocation_ordinal=0,
                    ),
                    TaskAllocationStep(1, TaskAllocationStepKind.RELEASE),
                ),
            ),
        ),
    )

    replay = replay_admission(
        program,
        schedule,
        topology=topology,
    )

    assert replay.pool.peak_allocated_bytes == 32
    assert replay.pool.final_allocated_bytes == 0
    assert tuple(step.operation.kind for step in replay.operations) == (
        AdmissionReplayOperationKind.ACQUIRE,
        AdmissionReplayOperationKind.BEGIN_RETIREMENT,
        AdmissionReplayOperationKind.COMPLETE_RETIREMENT,
    )


def test_after_task_fetch_reserves_task_released_range_causally() -> None:
    program = _program(
        (
            AliasGroupSpec("released", "cuda_0", 64),
            AliasGroupSpec("fetched", "cuda_0", 64),
        ),
        (
            ObjectSpec("released_object", "released", 0, 64),
            ObjectSpec("fetched_object", "fetched", 0, 64),
        ),
        (
            TaskSpec(
                "boundary",
                COMPUTE,
                "profile",
                inputs=("released_object",),
            ),
        ),
    )
    schedule = MemorySchedule(
        (
            ResidencySpec("released", MemoryLocation.DEVICE),
            ResidencySpec("fetched", MemoryLocation.SPILL),
        ),
        (
            MemoryAction("boundary", "released", MemoryActionKind.RELEASE),
            MemoryAction("boundary", "fetched", MemoryActionKind.PREFETCH),
        ),
        (ResidencySpec("fetched", MemoryLocation.DEVICE),),
    )

    replay = replay_admission(
        program,
        schedule,
        topology=_empty_topology(program, 64),
    )

    assert replay.pool.peak_allocated_bytes == 64
    assert replay.pool.final_allocated_bytes == 64
    assert len(replay.dependencies) == 1
    dependency = replay.dependencies[0]
    assert dependency.predecessor_purpose is AdmissionReplayPurpose.RELEASE
    assert dependency.predecessor_task_id == "boundary"
    assert dependency.predecessor_action_index == 0
    assert dependency.successor_action_index == 1
    reservation = next(
        decision
        for step, decision in zip(replay.operations, replay.pool.decisions, strict=True)
        if step.purpose is AdmissionReplayPurpose.FETCH_DESTINATION
        and step.operation.kind is AdmissionReplayOperationKind.RESERVE
    )
    assert reservation.physical_bytes_delta == 0


def test_admission_replay_emits_causal_eviction_to_fetch_dependency() -> None:
    program = _program(
        (AliasGroupSpec("state", "cuda_0", 96, retain_spill_copy=True),),
        (ObjectSpec("state_object", "state", 0, 96),),
        (
            TaskSpec("produce", COMPUTE, "profile", inputs=("state_object",)),
            TaskSpec("trigger", COMPUTE, "profile", dependencies=("produce",)),
            TaskSpec(
                "consume",
                COMPUTE,
                "profile",
                dependencies=("trigger",),
                inputs=("state_object",),
            ),
        ),
    )
    schedule = MemorySchedule(
        (ResidencySpec("state", MemoryLocation.DEVICE),),
        (
            MemoryAction("produce", "state", MemoryActionKind.OFFLOAD),
            MemoryAction("trigger", "state", MemoryActionKind.PREFETCH),
        ),
        (ResidencySpec("state", MemoryLocation.DEVICE),),
    )

    replay = replay_admission(
        program,
        schedule,
        topology=_empty_topology(program, 96),
    )

    assert replay.pool.peak_allocated_bytes == 96
    assert replay.pool.final_allocated_bytes == 96
    assert len(replay.dependencies) == 1
    dependency = replay.dependencies[0]
    assert dependency.predecessor_task_id == "produce"
    assert dependency.predecessor_alias_group_id == "state"
    assert dependency.predecessor_action_index == 0
    assert dependency.successor_task_id == "trigger"
    assert dependency.successor_alias_group_id == "state"
    assert dependency.successor_action_index == 1
    fetch_reservation = next(
        decision
        for step, decision in zip(replay.operations, replay.pool.decisions, strict=True)
        if step.purpose is AdmissionReplayPurpose.FETCH_DESTINATION
        and decision.requested_bytes
    )
    assert fetch_reservation.physical_bytes_delta == 0


def test_admission_replay_replaces_mutation_without_double_charging() -> None:
    program = _program(
        (AliasGroupSpec("weight", "cuda_0", 64),),
        (ObjectSpec("weight_object", "weight", 0, 64),),
        (
            TaskSpec(
                "update",
                COMPUTE,
                "profile",
                inputs=("weight_object",),
                mutations=(MutationSpec("weight_object"),),
            ),
        ),
        workspace_bytes=80,
    )
    schedule = MemorySchedule(
        (ResidencySpec("weight", MemoryLocation.DEVICE),),
        (),
        (ResidencySpec("weight", MemoryLocation.DEVICE),),
    )

    replay = replay_admission(
        program,
        schedule,
        topology=AdmissionTopology(
            "cuda_0",
            144,
            144,
            1,
            (
                _task_admission(
                    "update",
                    workspace_extents=(16,),
                    replacements=(("weight", 64),),
                ),
            ),
        ),
    )

    assert replay.workspace_bytes_by_task == (("update", 16),)
    assert replay.pool.peak_allocated_bytes == 144
    assert replay.pool.final_allocated_bytes == 64
    assert replay.ownership_transitions[0].kind is (
        OwnershipTransitionKind.MUTATION_REPLACEMENT
    )
    task_retirements = {
        step.purpose: step.operation.dependency_id
        for step in replay.operations
        if step.operation.kind is AdmissionReplayOperationKind.BEGIN_RETIREMENT
    }
    assert (
        task_retirements[AdmissionReplayPurpose.TASK_WORKSPACE]
        == (task_retirements[AdmissionReplayPurpose.MUTATION_REPLACEMENT])
    )


def test_admission_replay_handoff_changes_owner_without_allocating() -> None:
    program = _program(
        (
            AliasGroupSpec("residual", "cuda_0", 64),
            AliasGroupSpec("gradient", "cuda_0", 64),
        ),
        (
            ObjectSpec("residual_object", "residual", 0, 64),
            ObjectSpec("gradient_object", "gradient", 0, 64),
        ),
        (
            TaskSpec(
                "backward",
                COMPUTE,
                "profile",
                inputs=("residual_object",),
                outputs=("gradient_object",),
            ),
        ),
    )
    schedule = MemorySchedule(
        (ResidencySpec("residual", MemoryLocation.DEVICE),),
        (MemoryAction("backward", "residual", MemoryActionKind.RELEASE),),
        (ResidencySpec("gradient", MemoryLocation.DEVICE),),
    )

    replay = replay_admission(
        program,
        schedule,
        topology=AdmissionTopology(
            "cuda_0",
            64,
            64,
            1,
            (
                TaskAdmissionSpec(
                    "backward",
                    storage_handoffs=(StorageHandoff("residual", "gradient"),),
                ),
            ),
        ),
    )

    assert replay.pool.peak_allocated_bytes == 64
    assert replay.pool.final_allocated_bytes == 64
    assert replay.final_execution_aliases == ("gradient",)
    assert replay.ownership_transitions[0].kind is (
        OwnershipTransitionKind.STORAGE_HANDOFF
    )


def test_admission_replay_physical_decisions_ignore_task_timing() -> None:
    base = _program(
        (AliasGroupSpec("state", "cuda_0", 64),),
        (ObjectSpec("state_object", "state", 0, 64),),
        (TaskSpec("task", COMPUTE, "profile", inputs=("state_object",)),),
    )
    slower = replace(
        base,
        profiles=(replace(base.profiles[0], runtime_ns=1_000_000),),
    )
    schedule = MemorySchedule(
        (ResidencySpec("state", MemoryLocation.DEVICE),),
        (),
        (ResidencySpec("state", MemoryLocation.DEVICE),),
    )

    topology = _empty_topology(base, 64)
    first = replay_admission(
        base,
        schedule,
        topology=topology,
    )
    second = replay_admission(
        slower,
        schedule,
        topology=topology,
    )

    assert first.pool.decision_digest == second.pool.decision_digest
    assert first.pool.decisions == second.pool.decisions
