"""Standalone deterministic execution and memory simulation."""

from shadowspill.ir import MemorySchedule, Program, RecomputationSelection

from ._capi import load_simulator_library
from ._compiled import simulate_compiled
from ._python import simulate_python
from .model import (
    ActionPhysicalDelta,
    DeviceMemoryPeak,
    DeviceSimulationConfig,
    MemoryReuseDependency,
    MemorySnapshot,
    SimulationAdmission,
    SimulationConfig,
    SimulationInfeasibleError,
    SimulationResult,
    TaskInterval,
    TaskPhysicalDelta,
    TransferDirection,
    TransferInterval,
)


def simulate(
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[RecomputationSelection, ...] = (),
    config: SimulationConfig,
    admission: SimulationAdmission | None = None,
    record_timeline: bool = False,
) -> SimulationResult:
    """Replay an explicit memory schedule without invoking the planner.

    The compiled simulator is a required installed component.  A detailed
    Python timeline remains available through the explicit ``record_timeline``
    diagnostic mode, but missing or ABI-incompatible C code never causes an
    implicit fallback.
    """

    load_simulator_library()
    if not record_timeline:
        return simulate_compiled(
            program,
            schedule,
            selections=selections,
            config=config,
            admission=admission,
        )
    return simulate_python(
        program,
        schedule,
        selections=selections,
        config=config,
        admission=admission,
        record_timeline=record_timeline,
    )


__all__ = [
    "ActionPhysicalDelta",
    "DeviceMemoryPeak",
    "DeviceSimulationConfig",
    "MemoryReuseDependency",
    "MemorySnapshot",
    "SimulationAdmission",
    "SimulationConfig",
    "SimulationInfeasibleError",
    "SimulationResult",
    "TaskInterval",
    "TaskPhysicalDelta",
    "TransferDirection",
    "TransferInterval",
    "simulate",
]
