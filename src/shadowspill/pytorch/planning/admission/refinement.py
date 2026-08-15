"""Monotonic PressureFit refinement for fixed physical-layout feasibility."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from shadowspill.ir import MemoryLocation
from shadowspill.planner import AdmissionTopology, PressureFitResult
from shadowspill.planner._cache import CachedPressureFitResult
from shadowspill.simulator import SimulationConfig

from .layout import (
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
    build_fixed_layout_admission,
)

_MIB = 1 << 20
_FINE_STEP_BYTES = 128 * _MIB
_FINE_LIMIT_BYTES = 1 << 30
_COARSE_STEP_BYTES = 512 * _MIB


@dataclass(frozen=True, slots=True)
class FixedLayoutAttempt:
    """One deterministic physical-layout trial during capacity refinement."""

    requested_object_capacity_bytes: int
    effective_object_capacity_bytes: int
    required_bytes: int
    pool_capacity_bytes: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class FixedLayoutSelection:
    """One PressureFit selection and the exact physical certificate it passed."""

    pressurefit: CachedPressureFitResult
    topology: AdmissionTopology
    admission: FixedLayoutAdmission
    attempts: tuple[FixedLayoutAttempt, ...]
    original_object_capacity_bytes: int

    @property
    def result(self) -> PressureFitResult:
        return self.pressurefit.result

    @property
    def cache_hit(self) -> bool:
        return self.pressurefit.cache_hit

    @property
    def capacity_reduction_bytes(self) -> int:
        return (
            self.original_object_capacity_bytes
            - self.topology.object_capacity_bytes
        )


def resolve_fixed_layout_selection(
    config: SimulationConfig,
    topology: AdmissionTopology,
    resolve: Callable[[SimulationConfig], CachedPressureFitResult],
    *,
    progress: Callable[[str], None] | None = None,
) -> FixedLayoutSelection:
    """Select the highest-capacity PressureFit result with a valid layout.

    PressureFit's object capacity is reduced monotonically while the physical
    pool itself remains unchanged.  The returned record owns the exact
    effective topology and certificate; callers never reconstruct either from
    the original request.
    """

    original_capacity = _single_device_capacity(config, topology.device_id)
    attempts: list[FixedLayoutAttempt] = []
    last_error: FixedLayoutInfeasibleError | None = None
    for reduction in _capacity_reductions(original_capacity):
        requested_capacity = original_capacity - reduction
        requested_config = _config_with_capacity(
            config,
            device_id=topology.device_id,
            capacity_bytes=requested_capacity,
        )
        # PressureFit selects against logical capacity. The fixed-layout
        # builder below is the sole physical-placement authority for this
        # strategy; prefiltering through dynamic-pool admission would discard
        # schedules that are feasible under certified fixed placement.
        selected = resolve(requested_config)
        effective_topology = replace(
            topology,
            object_capacity_bytes=_effective_object_capacity(
                selected,
                requested_capacity=requested_capacity,
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
                effective_topology,
                dynamic_alias_group_ids=dynamic_aliases,
            )
        except FixedLayoutInfeasibleError as error:
            last_error = error
            attempts.append(
                FixedLayoutAttempt(
                    requested_capacity,
                    effective_topology.object_capacity_bytes,
                    error.required_bytes,
                    error.capacity_bytes,
                    False,
                )
            )
            if progress is not None:
                progress(
                    "fixed layout rejected PressureFit capacity "
                    f"{effective_topology.object_capacity_bytes}: "
                    f"required={error.required_bytes}, "
                    f"physical_pool={error.capacity_bytes}, "
                    f"requested_reduction={reduction}"
                )
            continue
        attempts.append(
            FixedLayoutAttempt(
                requested_capacity,
                effective_topology.object_capacity_bytes,
                admitted.layout.required_bytes,
                admitted.layout.pool_capacity_bytes,
                True,
            )
        )
        if progress is not None:
            progress(
                "fixed layout accepted PressureFit capacity "
                f"{effective_topology.object_capacity_bytes}: "
                f"fixed_slice={admitted.layout.fixed_slice_bytes}, "
                f"dynamic_reserve={admitted.layout.dynamic_reserve_bytes}, "
                f"slack={admitted.layout.slack_bytes}, "
                f"total_reduction="
                f"{original_capacity - effective_topology.object_capacity_bytes}"
            )
        return FixedLayoutSelection(
            selected,
            effective_topology,
            admitted,
            tuple(attempts),
            original_capacity,
        )
    if last_error is None:
        raise ValueError("fixed-layout refinement has no positive capacity")
    raise last_error


def _effective_object_capacity(
    selected: CachedPressureFitResult,
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


def _capacity_reductions(capacity_bytes: int) -> tuple[int, ...]:
    result = [0]
    reduction = _FINE_STEP_BYTES
    while reduction < capacity_bytes:
        result.append(reduction)
        reduction += (
            _FINE_STEP_BYTES if reduction < _FINE_LIMIT_BYTES else _COARSE_STEP_BYTES
        )
    return tuple(result)


def _single_device_capacity(config: SimulationConfig, device_id: str) -> int:
    matches = tuple(
        item.capacity_bytes for item in config.devices if item.device_id == device_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"simulation config must contain exactly one {device_id!r} device"
        )
    return matches[0]


def _config_with_capacity(
    config: SimulationConfig,
    *,
    device_id: str,
    capacity_bytes: int,
) -> SimulationConfig:
    return replace(
        config,
        devices=tuple(
            replace(item, capacity_bytes=capacity_bytes)
            if item.device_id == device_id
            else item
            for item in config.devices
        ),
    )


__all__ = [
    "FixedLayoutAttempt",
    "FixedLayoutSelection",
    "resolve_fixed_layout_selection",
]
