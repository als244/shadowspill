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
    relax_capacity: bool = False,
) -> SimulationResult:
    """Replay an explicit schedule through the simulator.

    `relax_capacity` drops device and spill capacity enforcement, so a
    schedule that overflows still reports how long it would take. A schedule
    that fits is unaffected, because its capacity checks never fired.
    """

    simulator_api()
    return simulate_program(
        program,
        schedule,
        selections=selections,
        config=config,
        admission=admission,
        relax_capacity=relax_capacity,
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
