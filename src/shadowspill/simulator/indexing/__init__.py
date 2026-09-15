"""The simulator's input in indexed form, and its result decoded back.

A template is the part that does not change when a schedule changes --
the program's geometry, indexed once -- so a caller pricing many
schedules for one program pays for it once. Binding a schedule onto a
template gives the simulator its complete input.

:mod:`.template` indexes the program, :mod:`.binding` binds one schedule onto
it, :mod:`.results` decodes what came back, and :mod:`.arrays` owns the
buffers all three hand to C. The two entry points are here.
"""

from shadowspill.ir import (
    MemorySchedule,
    ShadowSpillProgram,
    TaskAlternativeChoice,
)

from ..model import (
    SimulationAdmission,
    SimulationConfig,
    SimulationResult,
)
from .binding import _bind_schedule, _project
from .results import (
    IntervalArrays,
    _simulate_projection,
    interval_arrays_from_result,
)
from .template import IndexedSimulationTemplate, index_simulation_template

__all__ = [
    "IndexedSimulationTemplate",
    "IntervalArrays",
    "index_simulation_template",
    "interval_arrays_from_result",
    "simulate_program",
    "simulate_template",
]


def simulate_program(
    program: ShadowSpillProgram,
    schedule: MemorySchedule,
    *,
    selections: tuple[TaskAlternativeChoice, ...] = (),
    config: SimulationConfig,
    admission: SimulationAdmission | None = None,
) -> SimulationResult:
    """Replay an explicit schedule through the simulator.

    Exported as :func:`shadowspill.simulator.simulate`, which is the name a
    caller outside the package uses.
    """

    projection = _project(program, schedule, selections, config, admission)
    return _simulate_projection(projection, schedule)


def simulate_template(
    template: IndexedSimulationTemplate,
    schedule: MemorySchedule,
    *,
    admission: SimulationAdmission | None = None,
) -> SimulationResult:
    """Replay a validated schedule using cached indexed program geometry."""

    return _simulate_projection(_bind_schedule(template, schedule, admission), schedule)


__all__ = [
    "IndexedSimulationTemplate",
    "IntervalArrays",
    "index_simulation_template",
    "simulate_program",
    "simulate_template",
]
