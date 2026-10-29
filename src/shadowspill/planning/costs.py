"""Convert explicit capacities and host costs to the existing simulator IR."""

from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication import CommunicationInfo
from shadowspill.communication.memory import communication_overhead
from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import ShadowSpillProgram
from shadowspill.simulator import SimulationConfig

from .transfers import communication_transfers


def simulation_inputs(
    profiled: ProfiledProgram,
    budgets: MemoryBudgets,
    communication: CommunicationInfo,
    reserved_bytes: int = 0,
) -> tuple[ShadowSpillProgram, SimulationConfig]:
    program = profiled._costed_ir(communication)
    spec = profiled.program.parallelism
    world_size = 1 if spec is None else spec.world_size
    if len(program.devices) != 1:
        raise NotImplementedError(
            "offline planning currently requires one symmetric rank-local program"
        )
    if world_size > 1 and (
        not (
            profiled.program.bindings.get("kind") == "forward"
            or (
                profiled.program.bindings.get("kind") in {"forward_backward", "step"}
                and profiled.program.bindings.get("distributed_training_lowered")
                in {"gradients", "step"}
            )
        )
        or "weight_shards" not in profiled.program.bindings
    ):
        raise NotImplementedError(
            "distributed training transfer lowering is not implemented"
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
    ) != list(range(world_size)):
        raise ValueError("topology must contain exactly the declared logical ranks")
    communication_bytes, _ = communication_overhead(profiled.program, communication)
    fixed_bytes = profiled.fixed_device_bytes + communication_bytes + reserved_bytes
    usable = budgets.execution_bytes - fixed_bytes
    if usable <= 0:
        raise PlanInfeasibleError(
            "fixed device overhead exhausts the total GPU allowance",
            kind="fixed_overhead",
            required_bytes=fixed_bytes,
            capacity_bytes=budgets.execution_bytes,
        )

    # Explicit phases charge both search and exact simulation. Copy and
    # collective volumes remain distinct; none of this changes task profiling.
    program = communication_transfers(profiled.program, program, communication)
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
