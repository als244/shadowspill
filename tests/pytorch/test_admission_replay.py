from __future__ import annotations

from dataclasses import replace

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
    TaskProfile,
    TaskSpec,
)
from shadowspill.pytorch.planning.admission.admission_replay import (
    AdmissionReplayPurpose,
    OwnershipTransitionKind,
    replay_admission,
)
from shadowspill.pytorch.planning.admission.bindings import TaskOutputBinding
from tests.planner._examples import COMPUTE, DEVICE


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
        execution_pool_bytes=128,
        alignment=1,
    )

    assert replay.pool.peak_allocated_bytes == 128
    assert replay.pool.final_allocated_bytes == 96
    assert replay.workspace_bytes_by_task == (("forward", 32),)
    purposes = tuple(item.purpose for item in replay.operations)
    assert AdmissionReplayPurpose.TASK_WORKSPACE in purposes
    assert AdmissionReplayPurpose.TASK_OUTPUT in purposes


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
        execution_pool_bytes=96,
        alignment=1,
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
        for step, decision in zip(
            replay.operations, replay.pool.decisions, strict=True
        )
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
        execution_pool_bytes=144,
        output_bindings={
            "update": (TaskOutputBinding(0, "weight", replacement=True),)
        },
        alignment=1,
    )

    assert replay.workspace_bytes_by_task == (("update", 16),)
    assert replay.pool.peak_allocated_bytes == 144
    assert replay.pool.final_allocated_bytes == 64
    assert replay.ownership_transitions[0].kind is (
        OwnershipTransitionKind.MUTATION_REPLACEMENT
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
        execution_pool_bytes=64,
        output_bindings={
            "backward": (
                TaskOutputBinding(
                    0,
                    "gradient",
                    source_alias_group_id="residual",
                ),
            )
        },
        alignment=1,
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

    first = replay_admission(base, schedule, execution_pool_bytes=64, alignment=1)
    second = replay_admission(
        slower, schedule, execution_pool_bytes=64, alignment=1
    )

    assert first.pool.decision_digest == second.pool.decision_digest
    assert first.pool.decisions == second.pool.decisions
