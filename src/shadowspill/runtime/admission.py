"""Physical-budget admission independent of frameworks and device vendors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import lcm

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
    REUSE = "reuse"


@dataclass(frozen=True, slots=True)
class AllocationEvent:
    """One ordered allocation or free in an annotated slab timeline."""

    position: int
    allocation_id: str
    operation: AllocationOperation
    bytes: int
    alignment: int = 256
    planned: bool = False
    source_allocation_id: str | None = None

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
        if self.operation is AllocationOperation.REUSE:
            if not self.source_allocation_id:
                raise ValueError("reuse event requires a source allocation ID")
            if self.source_allocation_id == self.allocation_id:
                raise ValueError("reuse source and destination must differ")
            if self.planned:
                raise ValueError("planned allocations cannot reuse pending extents")
        elif self.source_allocation_id is not None:
            raise ValueError("only reuse events accept a source allocation ID")


@dataclass(frozen=True, slots=True)
class SlabReplay:
    """Deterministic spatial result for one complete allocation timeline."""

    slab_bytes: int
    peak_allocated_bytes: int
    peak_fragmentation_bytes: int
    final_allocated_bytes: int
    final_largest_free_range_bytes: int


@dataclass(frozen=True, slots=True)
class SlabPlacement:
    """One allocation identity's immutable offset in a static slab layout."""

    allocation_id: str
    offset: int
    bytes: int

    def __post_init__(self) -> None:
        if not self.allocation_id:
            raise ValueError("slab placement ID must be non-empty")
        if self.offset < 0:
            raise ValueError("slab placement offset must be non-negative")
        if self.bytes <= 0:
            raise ValueError("slab placement bytes must be positive")


@dataclass(frozen=True, slots=True)
class SlabLayout:
    """Deterministic offline offsets plus their exact replay statistics."""

    replay: SlabReplay
    placements: tuple[SlabPlacement, ...]
    layout_bytes: int
    static_layout_bytes: int
    dynamic_allocation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layout_bytes < 0 or self.layout_bytes > self.replay.slab_bytes:
            raise ValueError("slab layout requirement exceeds its slab")
        if self.static_layout_bytes < 0 or self.static_layout_bytes > self.layout_bytes:
            raise ValueError("static slab layout height is invalid")

    def offset_by_allocation(self) -> dict[str, int]:
        """Return a fresh lookup table for callers constructing runtime records."""

        return {item.allocation_id: item.offset for item in self.placements}


@dataclass(slots=True)
class _AllocationLifetime:
    bytes: int
    alignment: int
    start: int
    end: int = 0
    identities: list[str] = field(default_factory=list)
    offset: int = 0

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    """Conservative v1 leeway, kept explicit for reports and tests."""

    minimum_provider_headroom_bytes: int = 1280 * MIB
    provider_growth_margin_bytes: int = 64 * MIB
    provider_granularity_bytes: int = 64 * MIB
    minimum_workspace_reserve_bytes: int = 512 * MIB
    workspace_numerator: int = 5
    workspace_denominator: int = 4
    workspace_granularity_bytes: int = 2 * MIB
    minimum_spill_leeway_bytes: int = 256 * MIB
    spill_leeway_percent: int = 10
    spill_granularity_bytes: int = 64 << 10

    def __post_init__(self) -> None:
        values = (
            self.minimum_provider_headroom_bytes,
            self.provider_growth_margin_bytes,
            self.minimum_workspace_reserve_bytes,
            self.minimum_spill_leeway_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("admission margins must be non-negative")
        positive = (
            self.provider_granularity_bytes,
            self.workspace_numerator,
            self.workspace_denominator,
            self.workspace_granularity_bytes,
            self.spill_granularity_bytes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("admission ratios and granularities must be positive")
        if self.spill_leeway_percent < 0:
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
        live_lease_evidence: tuple[tuple[int, int, int, int, int], ...] = (),
        free_range_evidence: tuple[tuple[int, int, str | None, str | None], ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.position = position
        self.free_bytes = free_bytes
        self.largest_free_range_bytes = largest_free_range_bytes
        self.live_lease_evidence = live_lease_evidence
        self.free_range_evidence = free_range_evidence


def _free_range_evidence(
    ranges: list[tuple[int, int]],
    live: dict[str, tuple[int, int]],
) -> tuple[tuple[int, int, str | None, str | None], ...]:
    """Describe the live allocations immediately bordering largest holes."""

    live_by_offset = sorted(
        (offset, bytes_, allocation_id)
        for allocation_id, (offset, bytes_) in live.items()
    )
    evidence: list[tuple[int, int, str | None, str | None]] = []
    for offset, bytes_ in sorted(ranges, key=lambda item: (-item[1], item[0]))[:4]:
        end = offset + bytes_
        left = next(
            (
                allocation_id
                for live_offset, live_bytes, allocation_id in reversed(live_by_offset)
                if live_offset + live_bytes == offset
            ),
            None,
        )
        right = next(
            (
                allocation_id
                for live_offset, _live_bytes, allocation_id in live_by_offset
                if live_offset == end
            ),
            None,
        )
        evidence.append((offset, bytes_, left, right))
    return tuple(evidence)


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
            candidates: list[tuple[int, int, int, int]] = []
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
                    candidates.append((aligned, index, leading, available))
            if not candidates:
                free_bytes = sum(bytes_ for _, bytes_ in ranges)
                largest = max((bytes_ for _, bytes_ in ranges), default=0)
                evidence = _free_range_evidence(ranges, live)
                raise AdmissionError(
                    f"allocation {event.allocation_id!r} needs {event.bytes} bytes "
                    f"at position {event.position}, but the slab has {free_bytes} "
                    f"free bytes and a {largest}-byte largest range; "
                    f"largest holes (offset, bytes, left, right)={evidence}",
                    kind="slab_fragmentation",
                    required_bytes=event.bytes,
                    capacity_bytes=slab_bytes,
                    position=event.position,
                    free_bytes=free_bytes,
                    largest_free_range_bytes=largest,
                    free_range_evidence=evidence,
                )
            if event.planned:
                aligned, index, leading, _available = min(
                    candidates,
                    key=lambda candidate: (candidate[3], candidate[0]),
                )
            else:
                aligned, index, leading, _available = min(
                    candidates,
                    key=lambda candidate: (candidate[3], -candidate[0]),
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
        elif event.operation is AllocationOperation.REUSE:
            if event.allocation_id in live:
                raise ValueError(f"allocation {event.allocation_id!r} is already live")
            source = live.pop(event.source_allocation_id or "", None)
            if source is None:
                raise ValueError(
                    f"reuse source {event.source_allocation_id!r} is not live"
                )
            if source[1] != event.bytes:
                raise ValueError(
                    f"reuse for {event.allocation_id!r} has {event.bytes} bytes; "
                    f"source has {source[1]}"
                )
            live[event.allocation_id] = source
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


def plan_slab_layout(
    slab_bytes: int,
    events: tuple[AllocationEvent, ...],
    *,
    dynamic_allocation_ids: frozenset[str] = frozenset(),
) -> SlabLayout:
    """Assign one deterministic address to every complete allocation lifetime.

    Online best-fit can reject an otherwise feasible interval set because an
    early choice fragments a later large request. This routine first observes
    the complete causal lifetime stream, packs the largest/longest intervals
    first, and then validates every allocation, reuse, and free against those
    immutable offsets. It never changes event ordering or object lifetimes.
    """

    if slab_bytes < 0:
        raise ValueError("slab bytes must be non-negative")
    lifetimes, lifetime_by_identity = _allocation_lifetimes(events)
    unknown_dynamic = dynamic_allocation_ids.difference(lifetime_by_identity)
    if unknown_dynamic:
        raise ValueError(
            "dynamic slab identities are absent from the timeline: "
            f"{sorted(unknown_dynamic)!r}"
        )
    dynamic_lifetimes = {
        id(lifetime): lifetime
        for identity in dynamic_allocation_ids
        for lifetime in (lifetime_by_identity[identity],)
    }
    dynamic = list(dynamic_lifetimes.values())
    static = [item for item in lifetimes if id(item) not in dynamic_lifetimes]
    dynamic_bytes = _pack_lifetimes(dynamic)
    static_bytes = _pack_lifetimes(static)
    dynamic_alignment = 1
    for lifetime in dynamic:
        dynamic_alignment = lcm(dynamic_alignment, lifetime.alignment)
    dynamic_base = (
        0
        if not dynamic
        else ((slab_bytes - dynamic_bytes) // dynamic_alignment) * dynamic_alignment
    )
    static_boundary = (
        _round_up(static_bytes, dynamic_alignment) if dynamic else static_bytes
    )
    dynamic_reserve = slab_bytes - dynamic_base if dynamic else 0
    required = static_boundary + dynamic_reserve
    if required > slab_bytes:
        largest = max(lifetimes, key=lambda item: item.bytes, default=None)
        largest_id = None if largest is None else largest.identities[0]
        raise AdmissionError(
            "static slab layout needs "
            f"{required} bytes while capacity is {slab_bytes}; "
            f"largest_allocation={largest_id!r}",
            kind="slab_layout",
            required_bytes=required,
            capacity_bytes=slab_bytes,
        )
    for lifetime in dynamic:
        lifetime.offset += dynamic_base

    replay = _replay_static_layout(slab_bytes, events, lifetime_by_identity)
    placements = tuple(
        SlabPlacement(identity, lifetime.offset, lifetime.bytes)
        for lifetime in sorted(lifetimes, key=lambda item: item.start)
        for identity in lifetime.identities
    )
    expanded_dynamic = tuple(
        sorted(identity for item in dynamic for identity in item.identities)
    )
    return SlabLayout(
        replay,
        placements,
        required,
        static_boundary,
        expanded_dynamic,
    )


def _pack_lifetimes(lifetimes: list[_AllocationLifetime]) -> int:
    """Pack one independently reserved address region from offset zero."""

    placed: list[_AllocationLifetime] = []
    for lifetime in sorted(
        lifetimes,
        key=lambda item: (
            -item.bytes,
            -item.duration,
            item.start,
            item.identities[0],
        ),
    ):
        lifetime.offset = _lowest_available_offset(lifetime, placed)
        placed.append(lifetime)
    return max((item.offset + item.bytes for item in lifetimes), default=0)


def _allocation_lifetimes(
    events: tuple[AllocationEvent, ...],
) -> tuple[list[_AllocationLifetime], dict[str, _AllocationLifetime]]:
    previous_position = -1
    live: dict[str, _AllocationLifetime] = {}
    all_identities: set[str] = set()
    lifetimes: list[_AllocationLifetime] = []
    by_identity: dict[str, _AllocationLifetime] = {}
    for index, event in enumerate(events):
        if event.position < previous_position:
            raise ValueError("allocation timeline positions must be non-decreasing")
        previous_position = event.position
        if event.operation is AllocationOperation.ALLOCATE:
            if event.allocation_id in all_identities:
                raise ValueError(
                    "allocation identity appears more than once: "
                    f"{event.allocation_id!r}"
                )
            lifetime = _AllocationLifetime(
                event.bytes,
                event.alignment,
                index,
                identities=[event.allocation_id],
            )
            lifetimes.append(lifetime)
            live[event.allocation_id] = lifetime
            by_identity[event.allocation_id] = lifetime
            all_identities.add(event.allocation_id)
        elif event.operation is AllocationOperation.REUSE:
            if event.allocation_id in all_identities:
                raise ValueError(
                    "allocation identity appears more than once: "
                    f"{event.allocation_id!r}"
                )
            source_id = event.source_allocation_id or ""
            reused = live.pop(source_id) if source_id in live else None
            if reused is None:
                raise ValueError(f"reuse source {source_id!r} is not live")
            if reused.bytes != event.bytes:
                raise ValueError(
                    f"reuse for {event.allocation_id!r} has {event.bytes} bytes; "
                    f"source has {reused.bytes}"
                )
            reused.alignment = lcm(reused.alignment, event.alignment)
            reused.identities.append(event.allocation_id)
            live[event.allocation_id] = reused
            by_identity[event.allocation_id] = reused
            all_identities.add(event.allocation_id)
        else:
            released = (
                live.pop(event.allocation_id)
                if event.allocation_id in live
                else None
            )
            if released is None:
                raise ValueError(f"allocation {event.allocation_id!r} is not live")
            if released.bytes != event.bytes:
                raise ValueError(
                    f"free for {event.allocation_id!r} has {event.bytes} bytes; "
                    f"allocation has {released.bytes}"
                )
            released.end = index
    terminal = len(events)
    for lifetime in live.values():
        lifetime.end = terminal
    return lifetimes, by_identity


def _lifetimes_overlap(
    left: _AllocationLifetime, right: _AllocationLifetime
) -> bool:
    return left.start < right.end and right.start < left.end


def _lowest_available_offset(
    lifetime: _AllocationLifetime,
    placed: list[_AllocationLifetime],
) -> int:
    occupied = sorted(
        (
            item.offset,
            item.offset + item.bytes,
        )
        for item in placed
        if _lifetimes_overlap(lifetime, item)
    )
    candidate = _round_up(0, lifetime.alignment)
    occupied_end = 0
    for start, end in occupied:
        if end <= occupied_end:
            continue
        start = max(start, occupied_end)
        if candidate + lifetime.bytes <= start:
            return candidate
        occupied_end = max(occupied_end, end)
        candidate = _round_up(max(candidate, occupied_end), lifetime.alignment)
    return candidate


def _replay_static_layout(
    slab_bytes: int,
    events: tuple[AllocationEvent, ...],
    lifetime_by_identity: dict[str, _AllocationLifetime],
) -> SlabReplay:
    live: dict[str, tuple[int, int]] = {}
    allocated = 0
    peak_allocated = 0
    peak_fragmentation = 0
    for event in events:
        lifetime = lifetime_by_identity[event.allocation_id]
        if event.operation is AllocationOperation.ALLOCATE:
            _validate_static_allocation(live, event, lifetime.offset, slab_bytes)
            live[event.allocation_id] = (lifetime.offset, event.bytes)
            allocated += event.bytes
        elif event.operation is AllocationOperation.REUSE:
            source = live.pop(event.source_allocation_id or "", None)
            if source is None or source != (lifetime.offset, event.bytes):
                raise ValueError("static slab reuse changed its physical extent")
            live[event.allocation_id] = source
        else:
            allocation = live.pop(event.allocation_id, None)
            if allocation != (lifetime.offset, event.bytes):
                raise ValueError("static slab free changed its physical extent")
            allocated -= event.bytes
        peak_allocated = max(peak_allocated, allocated)
        free_ranges = _static_free_ranges(slab_bytes, live)
        free_bytes = slab_bytes - allocated
        largest = max((bytes_ for _, bytes_ in free_ranges), default=0)
        peak_fragmentation = max(peak_fragmentation, free_bytes - largest)
    final_ranges = _static_free_ranges(slab_bytes, live)
    return SlabReplay(
        slab_bytes=slab_bytes,
        peak_allocated_bytes=peak_allocated,
        peak_fragmentation_bytes=peak_fragmentation,
        final_allocated_bytes=allocated,
        final_largest_free_range_bytes=max(
            (bytes_ for _, bytes_ in final_ranges), default=0
        ),
    )


def _validate_static_allocation(
    live: dict[str, tuple[int, int]],
    event: AllocationEvent,
    offset: int,
    slab_bytes: int,
) -> None:
    if offset % event.alignment != 0:
        raise ValueError("static slab placement violates allocation alignment")
    if offset + event.bytes > slab_bytes:
        raise ValueError("static slab placement exceeds capacity")
    end = offset + event.bytes
    for allocation_id, (other_offset, other_bytes) in live.items():
        other_end = other_offset + other_bytes
        if offset < other_end and other_offset < end:
            raise ValueError(
                "static slab placements overlap while live: "
                f"{event.allocation_id!r} and {allocation_id!r}"
            )


def _static_free_ranges(
    slab_bytes: int, live: dict[str, tuple[int, int]]
) -> list[tuple[int, int]]:
    occupied = sorted(set(live.values()))
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for offset, bytes_ in occupied:
        if cursor < offset:
            ranges.append((cursor, offset - cursor))
        cursor = max(cursor, offset + bytes_)
    if cursor < slab_bytes:
        ranges.append((cursor, slab_bytes - cursor))
    return ranges


def admit_physical_budget(
    *,
    device_budget_bytes: int,
    spill_budget_bytes: int,
    context_bytes: int,
    observed_external_bytes: int,
    maximum_task_workspace_bytes: int,
    predicted_spill_peak_bytes: int,
    allocation_timeline: tuple[AllocationEvent, ...] = (),
    policy: AdmissionPolicy | None = None,
) -> tuple[PhysicalAdmission, SlabReplay]:
    """Compute explicit reserves and spatially validate the physical slab."""

    policy = policy or AdmissionPolicy()
    inputs = (
        device_budget_bytes,
        spill_budget_bytes,
        context_bytes,
        observed_external_bytes,
        maximum_task_workspace_bytes,
        predicted_spill_peak_bytes,
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
    workspace_reserve = workspace_reserve_bytes(
        maximum_task_workspace_bytes, policy=policy
    )
    if workspace_reserve > slab_bytes:
        raise AdmissionError(
            "workspace reserve exceeds the admitted slab",
            kind="workspace_budget",
            required_bytes=workspace_reserve,
            capacity_bytes=slab_bytes,
        )
    spill_percentage = (
        predicted_spill_peak_bytes * policy.spill_leeway_percent + 99
    ) // 100
    spill_leeway = max(policy.minimum_spill_leeway_bytes, spill_percentage)
    spill_reservation = _round_up(
        predicted_spill_peak_bytes + spill_leeway,
        policy.spill_granularity_bytes,
    )
    if spill_reservation > spill_budget_bytes:
        raise AdmissionError(
            "host peak plus explicit leeway exceeds the host budget",
            kind="spill_budget",
            required_bytes=spill_reservation,
            capacity_bytes=spill_budget_bytes,
        )
    replay = replay_slab_timeline(slab_bytes, allocation_timeline)
    admission = PhysicalAdmission(
        device_budget_bytes=device_budget_bytes,
        spill_budget_bytes=spill_budget_bytes,
        context_bytes=context_bytes,
        provider_headroom_bytes=provider_headroom,
        slab_bytes=slab_bytes,
        workspace_reserve_bytes=workspace_reserve,
        spill_reservation_bytes=spill_reservation,
        predicted_fragmentation_bytes=replay.peak_fragmentation_bytes,
    )
    return admission, replay


def workspace_reserve_bytes(
    maximum_task_workspace_bytes: int,
    *,
    policy: AdmissionPolicy | None = None,
) -> int:
    """Return the conservative contiguous-workspace admission allowance."""

    if maximum_task_workspace_bytes < 0:
        raise ValueError("maximum task workspace must be non-negative")
    policy = policy or AdmissionPolicy()
    scaled = (
        maximum_task_workspace_bytes * policy.workspace_numerator
        + policy.workspace_denominator
        - 1
    ) // policy.workspace_denominator
    return max(
        policy.minimum_workspace_reserve_bytes,
        _round_up(scaled, policy.workspace_granularity_bytes),
    )
