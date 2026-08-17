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


def test_shared_policy_round_trips_and_is_indexed() -> None:
    program = _with_shared_weight(SharedResidencyPolicy.SHARED_READ_ONLY)

    restored = type(program).from_json(program.to_json())

    assert restored == program
    assert restored.alias_groups[1].shared_residency is (
        SharedResidencyPolicy.SHARED_READ_ONLY
    )


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


def test_shared_alias_cannot_be_replaced_or_recomputation_retained() -> None:
    program = _with_shared_weight(SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED)
    first = program.tasks[0]

    with pytest.raises(ValidationError, match="cannot be replaced"):
        replace(
            program,
            tasks=(replace(first, outputs=("weight",)), *program.tasks[1:]),
        )

    group = RecomputationGroup(
        "shared_retention",
        (RecomputationOption("invalid", (), ("weight_storage",)),),
    )
    with pytest.raises(ValidationError, match="runtime-resident"):
        replace(program, recomputation_groups=(group,))


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
                MemoryActionKind.OFFLOAD,
            ),
            MemoryAction(
                "backward_marker",
                "activation_storage",
                MemoryActionKind.PREFETCH,
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
