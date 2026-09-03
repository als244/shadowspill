from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    MutationSpec,
    RecomputationGroup,
    RecomputationOption,
    ResidencySpec,
    SharedResidencyPolicy,
    ValidationError,
    index_program,
)

from ._examples import SAVE_SELECTION, representative_program


def _with_shared_weight(policy: SharedResidencyPolicy):
    program = representative_program()
    aliases = tuple(
        replace(alias, shared_residency=policy)
        if alias.alias_group_id == "weight_storage"
        else alias
        for alias in program.alias_groups
    )
    return replace(program, alias_groups=aliases)


@pytest.mark.parametrize(
    ("policy", "expected_code"),
    [
        (SharedResidencyPolicy.SHARED_READ_ONLY, 1),
        (SharedResidencyPolicy.SHARED_WRITABLE_CAUSAL, 2),
        (SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED, 3),
    ],
)
def test_shared_policy_round_trips_and_is_indexed(
    policy: SharedResidencyPolicy,
    expected_code: int,
) -> None:
    program = _with_shared_weight(policy)

    restored = type(program).from_json(program.to_json())

    assert restored == program
    assert restored.alias_groups[1].shared_residency is policy
    assert index_program(program).alias_shared_residency[1] == expected_code


def test_shared_read_only_rejects_mutation() -> None:
    program = _with_shared_weight(SharedResidencyPolicy.SHARED_READ_ONLY)
    first = program.tasks[0]

    with pytest.raises(ValidationError, match="is read-only"):
        replace(
            program,
            tasks=(
                replace(first, mutations=(MutationSpec("weight"),)),
                *program.tasks[1:],
            ),
        )


def test_shared_writable_unordered_accepts_in_place_mutation() -> None:
    program = _with_shared_weight(SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED)
    first = program.tasks[0]

    mutated = replace(
        program,
        tasks=(
            replace(first, mutations=(MutationSpec("weight"),)),
            *program.tasks[1:],
        ),
    )

    assert mutated.tasks[0].mutations == (MutationSpec("weight"),)


def test_causal_shared_alias_can_be_published_but_not_recomputation_retained() -> None:
    program = _with_shared_weight(SharedResidencyPolicy.SHARED_WRITABLE_CAUSAL)
    first = program.tasks[0]

    published = replace(
        program,
        tasks=(
            replace(first, inputs=("input",), outputs=("weight",)),
            *program.tasks[1:2],
        ),
        recomputation_groups=(),
    )

    assert published.tasks[0].outputs == ("weight",)

    group = RecomputationGroup(
        "shared_retention",
        (RecomputationOption("invalid", (), ("weight_storage",)),),
    )
    with pytest.raises(ValidationError, match="runtime-resident"):
        replace(program, recomputation_groups=(group,))


@pytest.mark.parametrize(
    "policy",
    [
        SharedResidencyPolicy.SHARED_READ_ONLY,
        SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED,
    ],
)
def test_noncausal_shared_alias_cannot_publish_task_outputs(
    policy: SharedResidencyPolicy,
) -> None:
    program = _with_shared_weight(policy)
    first = program.tasks[0]

    with pytest.raises(ValidationError, match="cannot publish task outputs"):
        replace(
            program,
            tasks=(
                replace(first, inputs=("input",), outputs=("weight",)),
                *program.tasks[1:2],
            ),
            recomputation_groups=(),
        )


def test_schedule_treats_shared_input_as_runtime_resident() -> None:
    program = _with_shared_weight(SharedResidencyPolicy.SHARED_READ_ONLY)
    schedule = representative_program_schedule_without_shared_weight()

    schedule.validate(program, SAVE_SELECTION)


def representative_program_schedule_without_shared_weight() -> MemorySchedule:
    return MemorySchedule(
        initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction(
                "forward_save",
                "activation_storage",
                MemoryActionKind.EVICT,
            ),
            MemoryAction(
                "backward_marker",
                "activation_storage",
                MemoryActionKind.FETCH,
            ),
            MemoryAction(
                "consume",
                "activation_storage",
                MemoryActionKind.RELEASE,
            ),
        ),
        final_residency=(ResidencySpec("output_storage", MemoryLocation.DEVICE),),
    )


@pytest.mark.parametrize(
    ("schedule", "path"),
    [
        (
            MemorySchedule(
                initial_residency=(
                    ResidencySpec("weight_storage", MemoryLocation.DEVICE),
                ),
                actions=(),
            ),
            "schedule.initial_residency[0].alias_group_id",
        ),
        (
            MemorySchedule(
                initial_residency=(
                    ResidencySpec("input_storage", MemoryLocation.DEVICE),
                ),
                actions=(
                    MemoryAction(
                        "forward_save",
                        "weight_storage",
                        MemoryActionKind.RELEASE,
                    ),
                ),
            ),
            "schedule.actions[0].alias_group_id",
        ),
        (
            MemorySchedule(
                initial_residency=(
                    ResidencySpec("input_storage", MemoryLocation.DEVICE),
                ),
                actions=(),
                final_residency=(
                    ResidencySpec("weight_storage", MemoryLocation.DEVICE),
                ),
            ),
            "schedule.final_residency[0].alias_group_id",
        ),
    ],
)
def test_schedule_cannot_own_shared_residency(
    schedule: MemorySchedule,
    path: str,
) -> None:
    program = _with_shared_weight(SharedResidencyPolicy.SHARED_READ_ONLY)

    with pytest.raises(ValidationError) as caught:
        schedule.validate(program, SAVE_SELECTION)

    assert caught.value.path == path
