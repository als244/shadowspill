"""Readable orchestration for fixed execution-pool layout admission.

Admission answers two different questions, and they cost very different
amounts:

* *how many bytes would this schedule need?* — replay the schedule into
  leases, give each a lifetime, and place them. This is `measure_fixed_layout`.
* *is the resulting layout safe to run?* — recover the reuse dependencies that
  make shared offsets causally sound, and re-simulate against them. This is
  `certify_fixed_layout`.

A caller searching for a schedule that fits asks the first question many times
and the second once, on whatever it settles on, so the two are separate entry
points. `build_fixed_layout_admission` composes them for callers that only
want an admitted layout or an error.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.planner import AdmissionTopology, PressureFitResult
from shadowspill.simulator import (
    ActionPhysicalDelta,
    SimulationAdmission,
    TaskPhysicalDelta,
    simulate,
)

from ..admission_replay import (
    AdmissionReplayPurpose,
    AdmissionReplayStep,
    _AdmissionScriptBuilder,
)
from .dependencies import (
    recover_reuse_dependencies,
    simulator_reuse_dependencies,
)
from .lifetimes import build_lease_lifetimes
from .model import (
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
    FixedLayoutPlacement,
    FixedPhysicalLayout,
    LeaseLifetime,
)
from .placement import place_lifetimes


@dataclass(frozen=True, slots=True)
class FixedLayoutMeasurement:
    """What one schedule's layout would need, before it is certified.

    A measurement never rejects: a layout larger than the pool is reported
    through `fits` and `shortfall_bytes` so a caller can act on how far it
    missed by. `replay_operations` and `script_builder` carry the replay this
    was derived from, so certification does not repeat it.
    """

    required_bytes: int
    pool_capacity_bytes: int
    fixed_slice_bytes: int
    dynamic_reserve_bytes: int
    scratch_reserve_bytes: int
    placements: tuple[FixedLayoutPlacement, ...]
    dynamic_lifetimes: tuple[LeaseLifetime, ...]
    replay_operations: tuple[AdmissionReplayStep, ...]
    script_builder: _AdmissionScriptBuilder

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.pool_capacity_bytes

    @property
    def slack_bytes(self) -> int:
        """Pool bytes left unused. Negative when the layout does not fit."""

        return self.pool_capacity_bytes - self.required_bytes

    @property
    def shortfall_bytes(self) -> int:
        """Bytes by which the layout exceeds the pool, or zero when it fits."""

        return max(0, self.required_bytes - self.pool_capacity_bytes)


def measure_fixed_layout(
    selected: PressureFitResult,
    topology: AdmissionTopology,
    *,
    dynamic_alias_group_ids: frozenset[str] = frozenset(),
    scratch_reserve_bytes: int = 0,
) -> FixedLayoutMeasurement:
    """Place one schedule's leases and report the bytes it would need.

    Caller-owned outputs are deliberately excluded from the reusable fixed
    slice.  Their leases remain ordinary dynamic pool allocations so a caller
    may retain an output across later invocations without aliasing a planned
    address.  ``dynamic_alias_group_ids`` must therefore contain only terminal
    caller-owned aliases; all schedule-managed storage remains fixed.
    """

    topology.validate(selected.program)
    if scratch_reserve_bytes < 0:
        raise ValueError("dynamic scratch reserve must be non-negative")
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
    dynamic_lease_ids = _final_dynamic_lease_ids(
        builder,
        dynamic_alias_group_ids,
    )
    dynamic_lifetimes = tuple(
        item for item in lifetimes if item.lease_id in dynamic_lease_ids
    )
    _validate_dynamic_lifetimes(dynamic_lifetimes)
    fixed_lifetimes = tuple(
        item for item in lifetimes if item.lease_id not in dynamic_lease_ids
    )
    placements, fixed_slice_bytes = place_lifetimes(fixed_lifetimes)
    dynamic_reserve_bytes = sum(item.bytes for item in dynamic_lifetimes)
    return FixedLayoutMeasurement(
        required_bytes=(
            fixed_slice_bytes + dynamic_reserve_bytes + scratch_reserve_bytes
        ),
        pool_capacity_bytes=topology.pool_capacity_bytes,
        fixed_slice_bytes=fixed_slice_bytes,
        dynamic_reserve_bytes=dynamic_reserve_bytes,
        scratch_reserve_bytes=scratch_reserve_bytes,
        placements=placements,
        dynamic_lifetimes=dynamic_lifetimes,
        replay_operations=operations,
        script_builder=builder,
    )


def certify_fixed_layout(
    selected: PressureFitResult,
    topology: AdmissionTopology,
    measurement: FixedLayoutMeasurement,
) -> FixedLayoutAdmission:
    """Prove a measured layout is safe to run, and re-simulate against it.

    Two leases sharing an offset are only sound if the second cannot begin
    before the first has released it. Recovering those reuse dependencies and
    re-simulating under them is what turns a set of offsets into a certificate,
    and it is the half a search does not need until it has chosen.
    """

    if not measurement.fits:
        raise FixedLayoutInfeasibleError(
            measurement.required_bytes,
            measurement.pool_capacity_bytes,
        )
    builder = measurement.script_builder
    layout = FixedPhysicalLayout(
        program_digest=selected.program.digest,
        schedule_digest=selected.schedule.digest,
        topology_digest=topology.digest,
        pool_capacity_bytes=measurement.pool_capacity_bytes,
        fixed_slice_bytes=measurement.fixed_slice_bytes,
        dynamic_reserve_bytes=measurement.dynamic_reserve_bytes,
        scratch_reserve_bytes=measurement.scratch_reserve_bytes,
        required_bytes=measurement.required_bytes,
        placements=measurement.placements,
        reuse_dependencies=recover_reuse_dependencies(
            measurement.replay_operations, measurement.placements
        ),
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
            sorted(measurement.dynamic_lifetimes, key=lambda item: item.lease_id)
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


def build_fixed_layout_admission(
    selected: PressureFitResult,
    topology: AdmissionTopology,
    *,
    dynamic_alias_group_ids: frozenset[str] = frozenset(),
    scratch_reserve_bytes: int = 0,
) -> FixedLayoutAdmission:
    """Measure and certify one physical layout, or raise if it does not fit."""

    return certify_fixed_layout(
        selected,
        topology,
        measure_fixed_layout(
            selected,
            topology,
            dynamic_alias_group_ids=dynamic_alias_group_ids,
            scratch_reserve_bytes=scratch_reserve_bytes,
        ),
    )


def _final_dynamic_lease_ids(
    builder: _AdmissionScriptBuilder,
    alias_group_ids: frozenset[str],
) -> frozenset[int]:
    """Resolve caller-owned aliases to their final physical generations.

    An alias may be produced, evicted, and fetched before caller handoff.  Only
    the lease active at the final boundary escapes the reusable fixed slice;
    historical generations remain ordinary fixed lifetimes.
    """

    missing = sorted(alias_group_ids - builder.active_aliases.keys())
    if missing:
        raise ValueError(
            "dynamic terminal aliases lack final execution leases: "
            f"{missing}"
        )
    return frozenset(builder.active_aliases[item] for item in alias_group_ids)


def _validate_dynamic_lifetimes(lifetimes: tuple[LeaseLifetime, ...]) -> None:
    """Require each dynamic exception to be one caller-owned final lease."""

    for item in lifetimes:
        if item.purpose not in {
            AdmissionReplayPurpose.TASK_OUTPUT,
            AdmissionReplayPurpose.FETCH_DESTINATION,
        }:
            raise ValueError(
                "only terminal caller-owned task outputs or their final fetch "
                "destinations may remain dynamic; "
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


__all__ = [
    "FixedLayoutMeasurement",
    "build_fixed_layout_admission",
    "certify_fixed_layout",
    "measure_fixed_layout",
]
