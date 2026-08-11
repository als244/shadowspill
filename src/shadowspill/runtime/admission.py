"""Physical-budget admission independent of frameworks and device vendors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shadowspill.ir import PhysicalAdmission

MIB = 1 << 20


def _round_up(value: int, granularity: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if granularity <= 0:
        raise ValueError("granularity must be positive")
    return ((value + granularity - 1) // granularity) * granularity


class AllocationOperation(StrEnum):
    ALLOCATE = "allocate"
    FREE = "free"


@dataclass(frozen=True, slots=True)
class AllocationEvent:
    """One ordered allocation or free in an annotated slab timeline."""

    position: int
    allocation_id: str
    operation: AllocationOperation
    bytes: int
    alignment: int = 256
    planned: bool = False

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("allocation event position must be non-negative")
        if not self.allocation_id:
            raise ValueError("allocation event ID must be non-empty")
        if not isinstance(self.operation, AllocationOperation):
            raise TypeError("allocation event operation must be AllocationOperation")
        if self.bytes <= 0:
            raise ValueError("allocation event bytes must be positive")
        if self.alignment <= 0:
            raise ValueError("allocation event alignment must be positive")
        if not isinstance(self.planned, bool):
            raise TypeError("allocation event planned flag must be boolean")


@dataclass(frozen=True, slots=True)
class SlabReplay:
    """Deterministic spatial result for one complete allocation timeline."""

    slab_bytes: int
    peak_allocated_bytes: int
    peak_fragmentation_bytes: int
    final_allocated_bytes: int
    final_largest_free_range_bytes: int


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    """Conservative v1 leeway, kept explicit for reports and tests."""

    minimum_provider_headroom_bytes: int = 512 * MIB
    provider_growth_margin_bytes: int = 64 * MIB
    provider_granularity_bytes: int = 64 * MIB
    minimum_workspace_reserve_bytes: int = 512 * MIB
    workspace_numerator: int = 5
    workspace_denominator: int = 4
    workspace_granularity_bytes: int = 2 * MIB
    minimum_host_leeway_bytes: int = 256 * MIB
    host_leeway_percent: int = 10
    host_granularity_bytes: int = 64 << 10

    def __post_init__(self) -> None:
        values = (
            self.minimum_provider_headroom_bytes,
            self.provider_growth_margin_bytes,
            self.minimum_workspace_reserve_bytes,
            self.minimum_host_leeway_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("admission margins must be non-negative")
        positive = (
            self.provider_granularity_bytes,
            self.workspace_numerator,
            self.workspace_denominator,
            self.workspace_granularity_bytes,
            self.host_granularity_bytes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("admission ratios and granularities must be positive")
        if self.host_leeway_percent < 0:
            raise ValueError("host leeway percent must be non-negative")


class AdmissionError(ValueError):
    """Physical admission failure with machine-readable capacity evidence."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        required_bytes: int,
        capacity_bytes: int,
        position: int | None = None,
        free_bytes: int | None = None,
        largest_free_range_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.position = position
        self.free_bytes = free_bytes
        self.largest_free_range_bytes = largest_free_range_bytes


def _insert_and_coalesce(
    ranges: list[tuple[int, int]], offset: int, bytes_: int
) -> None:
    ranges.append((offset, bytes_))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for current_offset, current_bytes in ranges:
        if merged and merged[-1][0] + merged[-1][1] == current_offset:
            previous_offset, previous_bytes = merged[-1]
            merged[-1] = (previous_offset, previous_bytes + current_bytes)
        else:
            merged.append((current_offset, current_bytes))
    ranges[:] = merged


def replay_slab_timeline(
    slab_bytes: int, events: tuple[AllocationEvent, ...]
) -> SlabReplay:
    """Replay the production two-ended policy and reject spatial infeasibility."""

    if slab_bytes < 0:
        raise ValueError("slab bytes must be non-negative")
    previous_position = -1
    ranges = [] if slab_bytes == 0 else [(0, slab_bytes)]
    live: dict[str, tuple[int, int]] = {}
    allocated = 0
    peak_allocated = 0
    peak_fragmentation = 0
    for event in events:
        if event.position < previous_position:
            raise ValueError("allocation timeline positions must be non-decreasing")
        previous_position = event.position
        if event.operation is AllocationOperation.ALLOCATE:
            if event.allocation_id in live:
                raise ValueError(f"allocation {event.allocation_id!r} is already live")
            candidates: list[tuple[int, int, int]] = []
            for index, (offset, available) in enumerate(ranges):
                if event.planned:
                    aligned = _round_up(offset, event.alignment)
                elif event.bytes <= available:
                    aligned = (
                        offset
                        + available
                        - event.bytes
                        - (offset + available - event.bytes) % event.alignment
                    )
                else:
                    continue
                leading = aligned - offset
                if leading <= available and event.bytes <= available - leading:
                    candidates.append((aligned, index, leading))
            if not candidates:
                free_bytes = sum(bytes_ for _, bytes_ in ranges)
                largest = max((bytes_ for _, bytes_ in ranges), default=0)
                raise AdmissionError(
                    f"allocation {event.allocation_id!r} needs {event.bytes} bytes "
                    f"at position {event.position}, but the slab has {free_bytes} "
                    f"free bytes and a {largest}-byte largest range",
                    kind="slab_fragmentation",
                    required_bytes=event.bytes,
                    capacity_bytes=slab_bytes,
                    position=event.position,
                    free_bytes=free_bytes,
                    largest_free_range_bytes=largest,
                )
            aligned, index, leading = (
                min(candidates) if event.planned else max(candidates)
            )
            offset, available = ranges.pop(index)
            trailing = available - leading - event.bytes
            if leading:
                ranges.append((offset, leading))
            if trailing:
                ranges.append((aligned + event.bytes, trailing))
            ranges.sort()
            live[event.allocation_id] = (aligned, event.bytes)
            allocated += event.bytes
            peak_allocated = max(peak_allocated, allocated)
        else:
            allocation = live.pop(event.allocation_id, None)
            if allocation is None:
                raise ValueError(f"allocation {event.allocation_id!r} is not live")
            offset, allocated_bytes = allocation
            if event.bytes != allocated_bytes:
                raise ValueError(
                    f"free for {event.allocation_id!r} has {event.bytes} bytes; "
                    f"allocation has {allocated_bytes}"
                )
            allocated -= allocated_bytes
            _insert_and_coalesce(ranges, offset, allocated_bytes)
        free_bytes = slab_bytes - allocated
        largest = max((bytes_ for _, bytes_ in ranges), default=0)
        peak_fragmentation = max(peak_fragmentation, free_bytes - largest)
    final_largest = max((bytes_ for _, bytes_ in ranges), default=0)
    return SlabReplay(
        slab_bytes=slab_bytes,
        peak_allocated_bytes=peak_allocated,
        peak_fragmentation_bytes=peak_fragmentation,
        final_allocated_bytes=allocated,
        final_largest_free_range_bytes=final_largest,
    )


def admit_physical_budget(
    *,
    device_budget_bytes: int,
    host_budget_bytes: int,
    context_bytes: int,
    observed_external_bytes: int,
    maximum_task_workspace_bytes: int,
    predicted_host_peak_bytes: int,
    allocation_timeline: tuple[AllocationEvent, ...] = (),
    policy: AdmissionPolicy | None = None,
) -> tuple[PhysicalAdmission, SlabReplay]:
    """Compute explicit reserves and spatially validate the physical slab."""

    policy = policy or AdmissionPolicy()
    inputs = (
        device_budget_bytes,
        host_budget_bytes,
        context_bytes,
        observed_external_bytes,
        maximum_task_workspace_bytes,
        predicted_host_peak_bytes,
    )
    if any(value < 0 for value in inputs):
        raise ValueError("physical admission inputs must be non-negative")
    provider_needed = observed_external_bytes + policy.provider_growth_margin_bytes
    provider_headroom = max(
        policy.minimum_provider_headroom_bytes,
        _round_up(provider_needed, policy.provider_granularity_bytes),
    )
    fixed_device_bytes = context_bytes + provider_headroom
    if fixed_device_bytes >= device_budget_bytes:
        raise AdmissionError(
            "context and provider headroom leave no device slab",
            kind="fixed_device_budget",
            required_bytes=fixed_device_bytes + 1,
            capacity_bytes=device_budget_bytes,
        )
    slab_bytes = device_budget_bytes - fixed_device_bytes
    scaled_workspace = (
        maximum_task_workspace_bytes * policy.workspace_numerator
        + policy.workspace_denominator
        - 1
    ) // policy.workspace_denominator
    workspace_reserve = max(
        policy.minimum_workspace_reserve_bytes,
        _round_up(scaled_workspace, policy.workspace_granularity_bytes),
    )
    if workspace_reserve > slab_bytes:
        raise AdmissionError(
            "workspace reserve exceeds the admitted slab",
            kind="workspace_budget",
            required_bytes=workspace_reserve,
            capacity_bytes=slab_bytes,
        )
    host_percentage = (
        predicted_host_peak_bytes * policy.host_leeway_percent + 99
    ) // 100
    host_leeway = max(policy.minimum_host_leeway_bytes, host_percentage)
    host_reservation = _round_up(
        predicted_host_peak_bytes + host_leeway,
        policy.host_granularity_bytes,
    )
    if host_reservation > host_budget_bytes:
        raise AdmissionError(
            "host peak plus explicit leeway exceeds the host budget",
            kind="host_budget",
            required_bytes=host_reservation,
            capacity_bytes=host_budget_bytes,
        )
    replay = replay_slab_timeline(slab_bytes, allocation_timeline)
    admission = PhysicalAdmission(
        device_budget_bytes=device_budget_bytes,
        host_budget_bytes=host_budget_bytes,
        context_bytes=context_bytes,
        provider_headroom_bytes=provider_headroom,
        slab_bytes=slab_bytes,
        workspace_reserve_bytes=workspace_reserve,
        host_reservation_bytes=host_reservation,
        predicted_fragmentation_bytes=replay.peak_fragmentation_bytes,
    )
    return admission, replay
