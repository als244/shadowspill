"""Certify the fixed physical layout of the plan the search placed."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from shadowspill.ir import MemoryLocation
from shadowspill.planner.admission import AdmissionFacts

# Straight at the modules rather than the package: `layout/__init__` also
# re-exports the runtime projection, which binds to the installed
# allocator. Checking a fixed layout is correct needs none of that.
from shadowspill.planner.admission.layout.build import build_fixed_layout_admission
from shadowspill.planner.admission.layout.model import (
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
)
from shadowspill.planner.diagnostics import PressureFitDiagnostics
from shadowspill.planner.plan_store import PlanLookup
from shadowspill.planner.result import PressureFitResult
from shadowspill.simulator import SimulationConfig


@dataclass(frozen=True, slots=True)
class FixedLayoutAttempt:
    """One deterministic physical-layout trial during capacity refinement."""

    requested_object_capacity_bytes: int
    effective_object_capacity_bytes: int
    required_bytes: int
    pool_capacity_bytes: int
    accepted: bool
    pressurefit_wall_time_ns: int = 0
    physical_admission_wall_time_ns: int = 0
    pressurefit_diagnostics: PressureFitDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class FixedLayoutSelection:
    """One PressureFit selection and the exact physical certificate it passed."""

    pressurefit: PlanLookup
    facts: AdmissionFacts
    admission: FixedLayoutAdmission
    attempts: tuple[FixedLayoutAttempt, ...]
    original_object_capacity_bytes: int

    @property
    def result(self) -> PressureFitResult:
        return self.pressurefit.result

    @property
    def from_store(self) -> bool:
        return self.pressurefit.from_store

    @property
    def capacity_reduction_bytes(self) -> int:
        return self.original_object_capacity_bytes - self.facts.object_capacity_bytes

    @property
    def pressurefit_wall_time_ns(self) -> int:
        """Cumulative PressureFit/cache-resolution time across refinements."""

        return sum(item.pressurefit_wall_time_ns for item in self.attempts)

    @property
    def physical_admission_wall_time_ns(self) -> int:
        """Cumulative physical-layout construction time across refinements."""

        return sum(item.physical_admission_wall_time_ns for item in self.attempts)


def placement_facts(
    facts: AdmissionFacts,
    *,
    scratch_reserve_bytes: int,
) -> AdmissionFacts:
    """The pool as the search may measure a layout against it.

    The scratch reserve is taken off here because the search cannot know it:
    it is a frontend quantity, and the certificate adds it to what a layout
    requires. Handing the search the room actually left for placement is what
    keeps its measurement and the certificate answering the same question.
    """

    return replace(
        facts,
        pool_capacity_bytes=max(0, facts.pool_capacity_bytes - scratch_reserve_bytes),
    )


def resolve_fixed_layout_selection(
    config: SimulationConfig,
    facts: AdmissionFacts,
    resolve: Callable[[SimulationConfig], PlanLookup],
    *,
    scratch_reserve_bytes: int = 0,
    progress: Callable[[str], None] | None = None,
) -> FixedLayoutSelection:
    """Certify the layout of the plan PressureFit selected.

    The search measures each candidate's layout against this pool and
    answers with a plan that fits, so there is one capacity to certify and
    no ladder to walk: a rejection here is a real disagreement between the
    search's measurement and the certificate, not a capacity to retry.

    The returned record owns the exact effective facts and certificate;
    callers never reconstruct either from the original request.
    """

    original_capacity = _single_device_capacity(config, facts.device_id)
    attempts: list[FixedLayoutAttempt] = []

    started = time.perf_counter_ns()
    selected = resolve(config)
    pressurefit_wall_time_ns = time.perf_counter_ns() - started

    admission_started = time.perf_counter_ns()
    effective_facts = replace(
        facts,
        object_capacity_bytes=_effective_object_capacity(
            selected,
            requested_capacity=original_capacity,
        ),
    )
    dynamic_aliases = frozenset(
        item.alias_group_id
        for item in selected.result.schedule.final_residency
        if item.location is MemoryLocation.DEVICE
    )
    try:
        admitted = build_fixed_layout_admission(
            selected.result,
            effective_facts,
            dynamic_alias_group_ids=dynamic_aliases,
            scratch_reserve_bytes=scratch_reserve_bytes,
        )
    except FixedLayoutInfeasibleError as error:
        attempts.append(
            FixedLayoutAttempt(
                original_capacity,
                effective_facts.object_capacity_bytes,
                error.required_bytes,
                error.capacity_bytes,
                False,
                pressurefit_wall_time_ns,
                time.perf_counter_ns() - admission_started,
                selected.result.diagnostics,
            )
        )
        if progress is not None:
            progress(
                "fixed layout rejected PressureFit capacity "
                f"{effective_facts.object_capacity_bytes}: "
                f"required={error.required_bytes}, "
                f"physical_pool={error.capacity_bytes}"
            )
        raise
    attempts.append(
        FixedLayoutAttempt(
            original_capacity,
            effective_facts.object_capacity_bytes,
            admitted.layout.required_bytes,
            admitted.layout.pool_capacity_bytes,
            True,
            pressurefit_wall_time_ns,
            time.perf_counter_ns() - admission_started,
            selected.result.diagnostics,
        )
    )
    if progress is not None:
        progress(
            "fixed layout accepted PressureFit capacity "
            f"{effective_facts.object_capacity_bytes}: "
            f"fixed_slice={admitted.layout.fixed_slice_bytes}, "
            f"dynamic_reserve={admitted.layout.dynamic_reserve_bytes}, "
            f"scratch_reserve={admitted.layout.scratch_reserve_bytes}, "
            f"slack={admitted.layout.slack_bytes}"
        )
    return FixedLayoutSelection(
        selected,
        effective_facts,
        admitted,
        tuple(attempts),
        original_capacity,
    )


def _effective_object_capacity(
    selected: PlanLookup,
    *,
    requested_capacity: int,
) -> int:
    effective = selected.result.diagnostics.effective_object_capacity_bytes
    if effective is None:
        return requested_capacity
    if not 0 < effective <= requested_capacity:
        raise ValueError(
            "PressureFit reported an invalid effective object capacity: "
            f"effective={effective}, requested={requested_capacity}"
        )
    return effective


def _single_device_capacity(config: SimulationConfig, device_id: str) -> int:
    matches = tuple(
        item.capacity_bytes for item in config.devices if item.device_id == device_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"simulation config must contain exactly one {device_id!r} device"
        )
    return matches[0]


__all__ = [
    "FixedLayoutAttempt",
    "FixedLayoutSelection",
    "resolve_fixed_layout_selection",
]
