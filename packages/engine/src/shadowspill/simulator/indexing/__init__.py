"""The simulator's input in indexed form, and its result decoded back.

A template is the part that does not change when a schedule changes --
the program's geometry, indexed once -- so a caller pricing many
schedules for one program pays for it once. Binding a schedule onto a
template gives the simulator its complete input.

:mod:`.template` indexes the program, :mod:`.binding` binds one schedule onto
it, :mod:`.results` decodes what came back, and :mod:`.arrays` owns the
buffers all three hand to C. The two entry points are here.
"""

import ctypes

from shadowspill.ir import (
    MemorySchedule,
    ShadowSpillProgram,
    TaskAlternativeChoice,
)
from shadowspill.ir.schedule import schedule_refusal

from ..capi import CScheduleCheck, simulator_api
from ..model import (
    SimulationAdmission,
    SimulationConfig,
    SimulationResult,
)
from .binding import _bind_schedule, _project, _Projection
from .results import (
    IntervalArrays,
    _simulate_projection,
    interval_arrays_from_result,
)
from .template import IndexedSimulationTemplate, index_simulation_template


def simulate_program(
    program: ShadowSpillProgram,
    schedule: MemorySchedule,
    *,
    selections: tuple[TaskAlternativeChoice, ...] = (),
    config: SimulationConfig,
    admission: SimulationAdmission | None = None,
) -> SimulationResult:
    """Replay one schedule through the simulator and return what it cost.

    The schedule is validated against the program first. A fetch or task
    launch that does not fit waits for room, so a plan that comes up short
    is slower rather than refused; its shortfalls are reported in the
    result. The simulator runs in the ShadowSpill library, which must be
    installed.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program the schedule belongs to.
    schedule : MemorySchedule
        The memory actions and residency boundaries to replay.
    selections : tuple of TaskAlternativeChoice, default ()
        The option each task-alternative group takes, which decides the
        tasks that run.
    config : SimulationConfig
        The machine: one device entry per program device, and spill
        capacity.
    admission : SimulationAdmission, optional
        Physical pool accounting to charge against capacity. None charges
        object and workspace bytes.

    Returns
    -------
    SimulationResult
        The makespan, task and copy intervals, memory peaks and capacity
        shortfalls.

    Raises
    ------
    ValidationError
        If the schedule or selections are invalid for this program.
    ValueError
        If ``config`` or ``admission`` names devices other than the program's,
        shared residency exceeds a capacity, or ``admission`` names a task or
        action the schedule lacks.
    SimulationInfeasibleError
        If the schedule cannot run to completion on this machine.
    """

    projection = _project(program, schedule, selections, config, admission)
    return _simulate_projection(projection, schedule)


def simulate_template(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
    *,
    admission: SimulationAdmission | None = None,
    checked: bool = False,
) -> SimulationResult:
    """Replay a schedule using cached indexed program geometry.

    `checked` first holds the schedule to the rules `MemorySchedule.validate`
    states, in the library on the same binding the replay uses, and raises the
    error that method would; otherwise the schedule must already be valid.
    """

    projection = _bind_schedule(template, schedule, admission)
    if checked:
        _check_projection(projection, template, schedule)
    return _simulate_projection(projection, schedule)


def check_schedule(
    template: IndexedSimulationTemplate, schedule: MemorySchedule
) -> None:
    """Hold a schedule to the rules `MemorySchedule.validate` states, in the
    library, raising the error that method would."""

    _check_projection(_bind_schedule(template, schedule, None), template, schedule)


def _check_projection(
    projection: _Projection,
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
) -> None:
    flags = template.alias_flags
    buffer = (ctypes.c_uint8 * max(1, len(flags))).from_buffer_copy(flags or b"\0")
    result = CScheduleCheck()
    status = simulator_api().shadowspill_check_schedule(
        ctypes.byref(projection.program), buffer, ctypes.byref(result)
    )
    if status != 0:
        raise RuntimeError(f"the schedule check failed with status {status}")
    if result.refusal != 0:
        raise schedule_refusal(
            int(result.refusal),
            int(result.index),
            template.alias_ids[result.alias]
            if result.alias < len(template.alias_ids)
            else "",
            template.task_ids,
            schedule,
        )


__all__ = [
    "IndexedSimulationTemplate",
    "IntervalArrays",
    "check_schedule",
    "index_simulation_template",
    "interval_arrays_from_result",
    "simulate_program",
    "simulate_template",
]
