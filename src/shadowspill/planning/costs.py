"""Convert explicit capacities and host costs to the existing simulator IR."""

from dataclasses import replace

from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication import CommunicationInfo
from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import ShadowSpillProgram, TransferPhase, TransferSpec
from shadowspill.simulator import SimulationConfig

from .budgets import MemoryBudgets


def simulation_inputs(
    profiled: ProfiledProgram,
    budgets: MemoryBudgets,
    communication: CommunicationInfo,
) -> tuple[ShadowSpillProgram, SimulationConfig]:
    program = profiled._costed_ir()
    if len(program.devices) != 1:
        raise NotImplementedError(
            "the offline milestone currently supports one logical rank"
        )
    if any(
        alias.shared_residency is not None or alias.transfer is not None
        for alias in program.alias_groups
    ):
        raise NotImplementedError(
            "shared resources and distributed transfers migrate later"
        )
    if communication.topology and sorted(
        rank for group in communication.topology.values() for rank in group
    ) != [0]:
        raise ValueError("single-rank topology must contain logical rank 0 only")
    usable = budgets.execution_bytes - profiled.fixed_device_bytes
    if usable <= 0:
        raise PlanInfeasibleError(
            "fixed device overhead exhausts the total GPU allowance",
            kind="fixed_overhead",
            required_bytes=profiled.fixed_device_bytes,
            capacity_bytes=budgets.execution_bytes,
        )

    # The old simulator config's latency fields are metadata only. Explicit
    # phases charge latency in both the search heuristic and exact simulation.
    def phases(direction: str, size: int) -> tuple[TransferPhase, ...]:
        latency = round(getattr(communication, f"{direction}_latency_seconds") * 1e9)
        return (
            ()
            if not latency
            else (
                TransferPhase(
                    "timed",
                    duration_ns=latency,
                    resource=f"host_{direction}",
                    timing_class="host_latency",
                ),
            )
        ) + (TransferPhase("copy", bytes=size),)

    if communication.fetch_latency_seconds or communication.evict_latency_seconds:
        program = replace(
            program,
            alias_groups=tuple(
                replace(
                    alias,
                    transfer=TransferSpec(
                        alias.size_bytes,
                        phases("fetch", alias.size_bytes),
                        phases("evict", alias.size_bytes),
                    ),
                )
                for alias in program.alias_groups
            ),
        )
    rates = {}
    for direction in ("fetch", "evict"):
        rate = getattr(communication, f"{direction}_bytes_per_second")
        concurrent = getattr(communication, f"{direction}_concurrent_bytes_per_second")
        rates[f"{direction}_solo_bandwidth_bytes_per_second"] = round(rate)
        rates[f"{direction}_concurrent_bandwidth_bytes_per_second"] = round(
            rate if concurrent is None else concurrent
        )
    return program, SimulationConfig.single_device(
        program.devices[0].device_id,
        device_capacity_bytes=usable,
        spill_capacity_bytes=budgets.spill_bytes,
        **rates,
    )
