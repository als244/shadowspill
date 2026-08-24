"""Standalone deterministic execution and memory simulation."""

from shadowspill.ir import MemorySchedule, Program, RecomputationSelection

from .capi import simulator_api
from .indexed import simulate_program
from .model import (
    ActionPhysicalDelta,
    CapacityViolation,
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
    """Replay an explicit schedule through the simulator."""

    simulator_api()
    return simulate_program(
        program,
        schedule,
        selections=selections,
        config=config,
        admission=admission,
    )


__all__ = [
    "ActionPhysicalDelta",
    "CapacityViolation",
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
