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

from dataclasses import dataclass, replace

from shadowspill.planner import AdmissionFacts, PressureFitResult
from shadowspill.planner.admission.indexed import encode_schedule
from shadowspill.planner.admission.operations import (
    AdmissionOperations,
    build_admission_operations,
)
from shadowspill.planner.admission.placement import place_records
from shadowspill.simulator import (
    ActionPhysicalDelta,
    SimulationAdmission,
    TaskPhysicalDelta,
    simulate,
)

from ..admission_replay import AdmissionReplayPurpose
from ..setup import AdmissionSetup, build_admission_setup
from .dependencies import (
    recover_reuse_dependencies,
    simulator_reuse_dependencies,
)
from .lifetimes import LeaseLayout, resolve_lease_lifetimes
from .model import (
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
    FixedPhysicalLayout,
    LeaseLifetime,
)


@dataclass(frozen=True, slots=True)
class FixedLayoutMeasurement:
    """What one schedule's layout would need, before it is certified.

    A measurement never rejects: a layout larger than the pool is reported
    through `fits` and `shortfall_bytes` so a caller can act on how far it
    missed by. `layout` and `offsets` carry what this was derived from, still
    as the library's own arrays, so certification neither repeats the walk nor
    pays to name leases a rejected measurement will discard.
    """

    required_bytes: int
    pool_capacity_bytes: int
    fixed_slice_bytes: int
    dynamic_reserve_bytes: int
    scratch_reserve_bytes: int
    offsets: tuple[int, ...]
    layout: LeaseLayout
    operations: AdmissionOperations
    setup: AdmissionSetup

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
    facts: AdmissionFacts,
    *,
    dynamic_alias_group_ids: frozenset[str] = frozenset(),
    scratch_reserve_bytes: int = 0,
    setup: AdmissionSetup | None = None,
) -> FixedLayoutMeasurement:
    """Place one schedule's leases and report the bytes it would need.

    `setup` is the schedule-invariant half, which a caller measuring many
    schedules for one resolved program should build once and pass in. Omitted,
    it is built here.

    Caller-owned outputs are deliberately excluded from the reusable fixed
    slice.  Their leases remain ordinary dynamic pool allocations so a caller
    may retain an output across later invocations without aliasing a planned
    address.  ``dynamic_alias_group_ids`` must therefore contain only terminal
    caller-owned aliases; all schedule-managed storage remains fixed.
    """

    if scratch_reserve_bytes < 0:
        raise ValueError("dynamic scratch reserve must be non-negative")
    if setup is None:
        setup = build_admission_setup(
            selected.program,
            selected.selections,
            selected.simulation_config,
            facts,
        )
    encoded = encode_schedule(selected.schedule, setup.template)
    setup = replace(setup, action_trigger_tasks=encoded.action_trigger_tasks)
    operations = build_admission_operations(
        setup.template, setup.indexed_facts, encoded
    )
    layout = resolve_lease_lifetimes(
        operations, setup, selected.simulation, dynamic_alias_group_ids
    )
    _validate_dynamic_lifetimes(layout.dynamic_lifetimes)
    offsets, fixed_slice_bytes = place_records(
        layout.leases.lifetimes, layout.fixed_count
    )
    dynamic_reserve_bytes = sum(item.bytes for item in layout.dynamic_lifetimes)
    return FixedLayoutMeasurement(
        required_bytes=(
            fixed_slice_bytes + dynamic_reserve_bytes + scratch_reserve_bytes
        ),
        pool_capacity_bytes=facts.pool_capacity_bytes,
        fixed_slice_bytes=fixed_slice_bytes,
        dynamic_reserve_bytes=dynamic_reserve_bytes,
        scratch_reserve_bytes=scratch_reserve_bytes,
        offsets=offsets,
        layout=layout,
        operations=operations,
        setup=setup,
    )


def certify_fixed_layout(
    selected: PressureFitResult,
    facts: AdmissionFacts,
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
    resolved = measurement.layout
    placements = resolved.placements(measurement.offsets)
    layout = FixedPhysicalLayout(
        program_digest=selected.program.digest,
        schedule_digest=selected.schedule.digest,
        facts_digest=facts.digest,
        pool_capacity_bytes=measurement.pool_capacity_bytes,
        fixed_slice_bytes=measurement.fixed_slice_bytes,
        dynamic_reserve_bytes=measurement.dynamic_reserve_bytes,
        scratch_reserve_bytes=measurement.scratch_reserve_bytes,
        required_bytes=measurement.required_bytes,
        placements=placements,
        reuse_dependencies=recover_reuse_dependencies(
            measurement.operations, measurement.setup, placements
        ),
        initial_alias_leases=tuple(sorted(resolved.initial_alias_leases.items())),
        task_allocation_leases=tuple(
            (task_id, ordinal, lease_id)
            for (task_id, ordinal), lease_id in sorted(
                resolved.task_allocation_leases.items()
            )
        ),
        action_destination_leases=tuple(
            sorted(resolved.action_destination_leases.items())
        ),
        dynamic_lifetimes=tuple(
            sorted(resolved.dynamic_lifetimes, key=lambda item: item.lease_id)
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
    facts: AdmissionFacts,
    *,
    dynamic_alias_group_ids: frozenset[str] = frozenset(),
    scratch_reserve_bytes: int = 0,
    setup: AdmissionSetup | None = None,
) -> FixedLayoutAdmission:
    """Measure and certify one physical layout, or raise if it does not fit."""

    return certify_fixed_layout(
        selected,
        facts,
        measure_fixed_layout(
            selected,
            facts,
            dynamic_alias_group_ids=dynamic_alias_group_ids,
            scratch_reserve_bytes=scratch_reserve_bytes,
            setup=setup,
        ),
    )


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
