"""Standalone deterministic execution and memory simulation."""

from shadowspill.ir import MemorySchedule, Program, RecomputationSelection

from ._capi import load_simulator_library
from ._compiled import simulate_compiled
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
) -> SimulationResult:
    """Replay an explicit schedule through the required compiled simulator."""

    load_simulator_library()
    return simulate_compiled(
        program,
        schedule,
        selections=selections,
        config=config,
        admission=admission,
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
