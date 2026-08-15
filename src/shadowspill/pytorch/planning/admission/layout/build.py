"""Readable orchestration for fixed execution-pool layout admission."""

from __future__ import annotations

from shadowspill.planner import AdmissionTopology, PressureFitResult
from shadowspill.simulator import (
    ActionPhysicalDelta,
    SimulationAdmission,
    TaskPhysicalDelta,
    simulate,
)

from ..admission_replay import AdmissionReplayPurpose, _AdmissionScriptBuilder
from .dependencies import (
    recover_reuse_dependencies,
    simulator_reuse_dependencies,
)
from .lifetimes import build_lease_lifetimes
from .model import (
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
    FixedPhysicalLayout,
    LeaseLifetime,
)
from .placement import place_lifetimes


def build_fixed_layout_admission(
    selected: PressureFitResult,
    topology: AdmissionTopology,
    *,
    dynamic_alias_group_ids: frozenset[str] = frozenset(),
) -> FixedLayoutAdmission:
    """Build, causally certify, and re-simulate one physical layout.

    Caller-owned outputs are deliberately excluded from the reusable fixed
    slice.  Their leases remain ordinary dynamic pool allocations so a caller
    may retain an output across later invocations without aliasing a planned
    address.  ``dynamic_alias_group_ids`` must therefore contain only terminal
    caller-owned aliases; all schedule-managed storage remains fixed.
    """

    topology.validate(selected.program)
    builder = _AdmissionScriptBuilder(
        selected.program,
        selected.schedule,
        selected.selections,
        topology,
    )
    operations, *_ = builder.build()
    lifetimes = build_lease_lifetimes(
        operations,
        builder,
        selected.schedule,
        selected.simulation,
    )
    dynamic_lifetimes = tuple(
        item
        for item in lifetimes
        if item.alias_group_id in dynamic_alias_group_ids
    )
    _validate_dynamic_lifetimes(dynamic_lifetimes)
    dynamic_lease_ids = frozenset(item.lease_id for item in dynamic_lifetimes)
    fixed_lifetimes = tuple(
        item for item in lifetimes if item.lease_id not in dynamic_lease_ids
    )
    placements, fixed_slice_bytes = place_lifetimes(fixed_lifetimes)
    dynamic_reserve_bytes = sum(item.bytes for item in dynamic_lifetimes)
    required_bytes = fixed_slice_bytes + dynamic_reserve_bytes
    if required_bytes > topology.pool_capacity_bytes:
        raise FixedLayoutInfeasibleError(
            required_bytes,
            topology.pool_capacity_bytes,
        )
    dependencies = recover_reuse_dependencies(operations, placements)
    layout = FixedPhysicalLayout(
        program_digest=selected.program.digest,
        schedule_digest=selected.schedule.digest,
        topology_digest=topology.digest,
        pool_capacity_bytes=topology.pool_capacity_bytes,
        fixed_slice_bytes=fixed_slice_bytes,
        dynamic_reserve_bytes=dynamic_reserve_bytes,
        required_bytes=required_bytes,
        placements=placements,
        reuse_dependencies=dependencies,
        initial_alias_leases=tuple(sorted(builder.initial_alias_leases.items())),
        task_allocation_leases=tuple(
            (task_id, ordinal, lease_id)
            for (task_id, ordinal), lease_id in sorted(
                builder.task_allocation_leases.items()
            )
        ),
        action_destination_leases=tuple(
            sorted(builder.action_destination_leases.items())
        ),
        dynamic_lifetimes=tuple(
            sorted(dynamic_lifetimes, key=lambda item: item.lease_id)
        ),
    )
    simulator_input = _simulation_input(selected, layout)
    return FixedLayoutAdmission(
        layout=layout,
        simulator_input=simulator_input,
        simulation=simulate(
            selected.program,
            selected.schedule,
            selections=selected.selections,
            config=selected.simulation_config,
            admission=simulator_input,
        ),
    )


def _validate_dynamic_lifetimes(lifetimes: tuple[LeaseLifetime, ...]) -> None:
    """Require the dynamic exception to be a terminal task output only."""

    for item in lifetimes:
        if item.purpose is not AdmissionReplayPurpose.TASK_OUTPUT:
            raise ValueError(
                "only terminal caller-owned task outputs may remain dynamic; "
                f"lease {item.lease_id} has purpose {item.purpose.value!r}"
            )


def _simulation_input(
    selected: PressureFitResult,
    layout: FixedPhysicalLayout,
) -> SimulationAdmission:
    device_id = selected.program.devices[0].device_id
    tasks = selected.program.selected_tasks(selected.selections)
    return SimulationAdmission(
        initial_physical_bytes=((device_id, layout.required_bytes),),
        device_capacity_bytes=((device_id, layout.pool_capacity_bytes),),
        task_deltas=tuple(TaskPhysicalDelta(task.task_id, 0, 0) for task in tasks),
        action_deltas=tuple(
            ActionPhysicalDelta(index, 0, 0)
            for index in range(len(selected.schedule.actions))
        ),
        reuse_dependencies=simulator_reuse_dependencies(layout.reuse_dependencies),
    )


__all__ = ["build_fixed_layout_admission"]
