from __future__ import annotations

from shadowspill.ir import MemoryLocation, ResidencySpec

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


def test_recompute_selection_moves_the_active_writer() -> None:
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


def test_physical_admission_exposes_object_capacity() -> None:
    admission = representative_plan().admission

    assert admission.object_capacity_bytes == 768
