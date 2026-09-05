from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
    ValidationError,
)
from shadowspill.ir.schedule import first_use_initial_order

from ._examples import SAVE_SELECTION, representative_plan, representative_program


def test_save_selection_removes_inactive_task_and_dependencies() -> None:
    tasks = representative_program().selected_tasks(SAVE_SELECTION)

    assert tuple(task.task_id for task in tasks) == (
        "forward_save",
        "backward_marker",
        "consume",
    )
    assert tasks[1].dependencies == ("forward_save",)
    assert tasks[2].dependencies == ("forward_save", "backward_marker")


def test_graph_pair_selection_moves_the_active_writer() -> None:
    program = representative_program()
    selection_type = type(SAVE_SELECTION[0])
    tasks = program.selected_tasks(
        (selection_type("activation_tradeoff", "recompute"),)
    )

    assert tuple(task.task_id for task in tasks) == (
        "backward_marker",
        "forward_recompute",
        "consume",
    )
    assert tasks[0].dependencies == ()
    assert tasks[2].dependencies == (
        "backward_marker",
        "forward_recompute",
    )


def test_schedule_reaches_declared_final_residency() -> None:
    plan = representative_plan()
    plan.schedule.validate(plan.program, plan.selections)

    assert plan.schedule.final_residency == (
        ResidencySpec("output_storage", MemoryLocation.DEVICE),
    )


def _retained_spill_output_program() -> Program:
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("state_storage", "cuda_0", 64, retain_spill_copy=True),
        ),
        objects=(ObjectSpec("state", "state_storage", 0, 64),),
        profiles=(TaskProfile("update_profile", 10, 0, "update_abi"),),
        tasks=(
            TaskSpec(
                "update",
                ResourceSpec("cuda_0", ResourceKind.COMPUTE),
                "update_profile",
                outputs=("state",),
            ),
        ),
    )


def test_evict_refreshes_a_retained_spill_copy() -> None:
    program = _retained_spill_output_program()
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("state_storage", MemoryLocation.DEVICE),),
        actions=(MemoryAction("update", "state_storage", MemoryActionKind.EVICT),),
        final_residency=(ResidencySpec("state_storage", MemoryLocation.SPILL),),
    )

    schedule.validate(program)


def test_release_leaves_an_unchanged_retained_spill_copy_current() -> None:
    output_program = _retained_spill_output_program()
    program = replace(
        output_program,
        tasks=(replace(output_program.tasks[0], inputs=("state",), outputs=()),),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("state_storage", MemoryLocation.DEVICE),),
        actions=(MemoryAction("update", "state_storage", MemoryActionKind.RELEASE),),
        final_residency=(ResidencySpec("state_storage", MemoryLocation.SPILL),),
    )

    schedule.validate(program)


def test_output_invalidates_a_retained_spill_copy() -> None:
    program = _retained_spill_output_program()
    stale = MemorySchedule(
        initial_residency=(ResidencySpec("state_storage", MemoryLocation.DEVICE),),
        actions=(MemoryAction("update", "state_storage", MemoryActionKind.RELEASE),),
        final_residency=(ResidencySpec("state_storage", MemoryLocation.SPILL),),
    )

    with pytest.raises(ValidationError, match="current host residency"):
        stale.validate(program)


def test_first_use_initial_order_follows_the_task_sequence() -> None:
    program = representative_program()
    schedule = MemorySchedule(
        # Deliberately emitted against first-use order, with one alias no
        # task consumes as an input and one entry that is not on device.
        initial_residency=(
            ResidencySpec("activation_storage", MemoryLocation.DEVICE),
            ResidencySpec("output_storage", MemoryLocation.DEVICE),
            ResidencySpec("weight_storage", MemoryLocation.DEVICE),
            ResidencySpec("input_storage", MemoryLocation.SPILL),
        ),
        actions=(),
    )

    # forward_save consumes input then weight; consume reads the
    # activation later; nothing reads output, so it keeps its emitted
    # position after every consumed alias. The spilled entry never
    # appears.
    assert first_use_initial_order(program, schedule) == (
        "weight_storage",
        "activation_storage",
        "output_storage",
    )
