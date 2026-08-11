"""Standalone deterministic execution and memory simulation."""

from shadowspill.ir import MemorySchedule, Program, RecomputationSelection

from ._capi import simulator_library_path
from ._compiled import simulate_compiled
from ._python import simulate_python
from .model import (
    DeviceMemoryPeak,
    DeviceSimulationConfig,
    MemorySnapshot,
    SimulationConfig,
    SimulationInfeasibleError,
    SimulationResult,
    TaskInterval,
    TransferDirection,
    TransferInterval,
)


def simulate(
    program: Program,
    schedule: MemorySchedule,
    *,
    selections: tuple[RecomputationSelection, ...] = (),
    config: SimulationConfig,
    record_timeline: bool = False,
) -> SimulationResult:
    """Replay an explicit memory schedule without invoking the planner."""

    if not record_timeline and simulator_library_path() is not None:
        return simulate_compiled(
            program,
            schedule,
            selections=selections,
            config=config,
        )
    return simulate_python(
        program,
        schedule,
        selections=selections,
        config=config,
        record_timeline=record_timeline,
    )


__all__ = [
    "DeviceMemoryPeak",
    "DeviceSimulationConfig",
    "MemorySnapshot",
    "SimulationConfig",
    "SimulationInfeasibleError",
    "SimulationResult",
    "TaskInterval",
    "TransferDirection",
    "TransferInterval",
    "simulate",
]
