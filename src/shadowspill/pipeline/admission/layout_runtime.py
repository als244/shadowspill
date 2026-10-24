"""Project a semantic physical certificate into indexed runtime identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from shadowspill.ir import (
    MemoryActionKind,
    MemorySchedule,
    ShadowSpillProgram,
)
from shadowspill.ir.program import canonical_index
from shadowspill.ir.schedule import first_use_initial_order
from shadowspill.planner.admission.layout.model import (
    FixedLayoutPlacements,
    FixedPhysicalLayout,
    LeaseLifetime,
)
from shadowspill.runtime.admission_capi import NO_ID
from shadowspill.runtime.fixed_layout import (
    RuntimeFixedLayout,
    RuntimeFixedPlacement,
    RuntimePlacementKind,
)

from .layout_dependencies import ActionIdentity as _ActionIdentity
from .layout_dependencies import runtime_dependencies

#: Where one fixed lease lies: its offset, bytes and alignment.
_Fixed = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class DynamicTaskAllocationPolicy:
    """One task allocation served outside the reusable fixed-layout slice."""

    task_id: str
    allocation_ordinal: int
    bytes: int
    alignment: int


def project_runtime_fixed_layout(
    layout: FixedPhysicalLayout,
    program: ShadowSpillProgram,
    schedule: MemorySchedule,
    *,
    dynamic_task_allocations: tuple[DynamicTaskAllocationPolicy, ...] = (),
    anonymous_allocation_bytes: Mapping[tuple[str, int], int] | None = None,
) -> RuntimeFixedLayout:
    """Translate a layout, retaining its offsets, dependencies and reserved span.

    A shared distributed certificate can reserve larger temporary allocations
    than this rank's kernel uses. Bind those slots to the exact local size;
    every byte the runtime may use still lies within the certified range.
    """

    if layout.program_digest != program.digest:
        raise ValueError("fixed layout belongs to a different ShadowSpillProgram")
    if layout.schedule_digest != schedule.digest:
        raise ValueError("fixed layout belongs to a different memory schedule")

    fixed_by_lease = _fixed_by_lease(layout.placements)
    dynamic_by_lease = {item.lease_id: item for item in layout.dynamic_lifetimes}
    action_identities = _action_identities(program, schedule)
    task_allocations = _task_allocation_identities(layout)

    placements = [
        *_initial_placements(
            layout,
            program,
            schedule,
            fixed_by_lease,
        ),
        *_task_placements(
            layout,
            fixed_by_lease,
            dynamic_by_lease,
            dynamic_task_allocations=dynamic_task_allocations,
        ),
        *_action_placements(
            layout,
            fixed_by_lease,
            dynamic_by_lease,
            action_identities,
        ),
    ]
    if anonymous_allocation_bytes:
        sizes = {
            (canonical_index(task, "task_"), ordinal): size
            for (task, ordinal), size in anonymous_allocation_bytes.items()
        }
        for index, placement in enumerate(placements):
            if placement.kind is not RuntimePlacementKind.TASK_ALLOCATION:
                continue
            size = sizes.get((placement.task_id, placement.ordinal), placement.bytes)
            if not 0 < size <= placement.bytes:
                raise ValueError(
                    "local temporary allocation exceeds its certified slot"
                )
            if size != placement.bytes:
                placements[index] = replace(placement, bytes=size)
    dependencies = runtime_dependencies(
        layout,
        program,
        action_identities,
        task_allocations,
    )
    return RuntimeFixedLayout(
        slice_bytes=layout.fixed_slice_bytes,
        placements=tuple(
            sorted(
                placements,
                key=lambda item: (
                    int(item.kind),
                    item.task_id,
                    item.ordinal,
                    item.object_id,
                ),
            )
        ),
        dependencies=tuple(
            sorted(
                dependencies,
                key=lambda item: (
                    int(item.successor_kind),
                    item.successor_task_id,
                    item.successor_ordinal,
                    item.predecessor_task_id,
                    item.predecessor_action_ordinal,
                ),
            )
        ),
    )


def _initial_placements(
    layout: FixedPhysicalLayout,
    program: ShadowSpillProgram,
    schedule: MemorySchedule,
    fixed_by_lease: dict[int, _Fixed],
) -> tuple[RuntimeFixedPlacement, ...]:
    initial_leases = dict(layout.initial_alias_leases)
    sizes = {item.alias_group_id: item.size_bytes for item in program.alias_groups}
    # These values really are resident at entry. Fetch destinations are
    # represented separately by their ordinary scheduled action identities.
    aliases = tuple(
        alias_group_id
        for alias_group_id in first_use_initial_order(program, schedule)
        if sizes[alias_group_id] != 0
    )
    if set(aliases) != set(initial_leases):
        raise ValueError("fixed layout initial objects differ from the schedule")
    return tuple(
        _fixed_runtime_placement(
            fixed_by_lease[initial_leases[alias_id]],
            task_id=NO_ID,
            ordinal=NO_ID,
            object_id=canonical_index(alias_id, "alias_"),
            kind=RuntimePlacementKind.INITIAL_OBJECT,
        )
        for alias_id in aliases
    )


def _task_placements(
    layout: FixedPhysicalLayout,
    fixed_by_lease: dict[int, _Fixed],
    dynamic_by_lease: dict[int, LeaseLifetime],
    *,
    dynamic_task_allocations: tuple[DynamicTaskAllocationPolicy, ...],
) -> tuple[RuntimeFixedPlacement, ...]:
    result: list[RuntimeFixedPlacement] = []
    covered: set[tuple[str, int]] = set()
    for task_id, ordinal, lease_id in layout.task_allocation_leases:
        covered.add((task_id, ordinal))
        fixed = fixed_by_lease.get(lease_id)
        if fixed is not None:
            result.append(
                _fixed_runtime_placement(
                    fixed,
                    task_id=canonical_index(task_id, "task_"),
                    ordinal=ordinal,
                    object_id=NO_ID,
                    kind=RuntimePlacementKind.TASK_ALLOCATION,
                )
            )
            continue
        dynamic = dynamic_by_lease.get(lease_id)
        if dynamic is None:
            raise ValueError(f"task allocation lease {lease_id} has no policy")
        result.append(
            RuntimeFixedPlacement(
                task_id=canonical_index(task_id, "task_"),
                ordinal=ordinal,
                object_id=NO_ID,
                offset=NO_ID,
                bytes=dynamic.bytes,
                alignment=dynamic.alignment,
                kind=RuntimePlacementKind.DYNAMIC_TASK_ALLOCATION,
            )
        )
    for policy in dynamic_task_allocations:
        identity = policy.task_id, policy.allocation_ordinal
        if identity in covered:
            raise ValueError(
                f"task allocation {policy.task_id}:{policy.allocation_ordinal} "
                "has fixed and dynamic physical policies"
            )
        result.append(
            RuntimeFixedPlacement(
                task_id=canonical_index(policy.task_id, "task_"),
                ordinal=policy.allocation_ordinal,
                object_id=NO_ID,
                offset=NO_ID,
                bytes=policy.bytes,
                alignment=policy.alignment,
                kind=RuntimePlacementKind.DYNAMIC_TASK_ALLOCATION,
            )
        )
    return tuple(result)


def _action_placements(
    layout: FixedPhysicalLayout,
    fixed_by_lease: dict[int, _Fixed],
    dynamic_by_lease: dict[int, LeaseLifetime],
    action_identities: dict[int, _ActionIdentity],
) -> tuple[RuntimeFixedPlacement, ...]:
    result: list[RuntimeFixedPlacement] = []
    for action_index, lease_id in layout.action_destination_leases:
        try:
            identity = action_identities[action_index]
        except KeyError as error:
            raise ValueError(
                f"action destination {action_index} has no runtime identity"
            ) from error
        fixed = fixed_by_lease.get(lease_id)
        if fixed is not None:
            result.append(
                _fixed_runtime_placement(
                    fixed,
                    task_id=identity.task_id,
                    ordinal=identity.ordinal,
                    object_id=identity.object_id,
                    kind=RuntimePlacementKind.ACTION_DESTINATION,
                )
            )
            continue
        dynamic = dynamic_by_lease.get(lease_id)
        if dynamic is None:
            raise ValueError(
                f"action destination lease {lease_id} has no physical policy"
            )
        result.append(
            RuntimeFixedPlacement(
                task_id=identity.task_id,
                ordinal=identity.ordinal,
                object_id=identity.object_id,
                offset=NO_ID,
                bytes=dynamic.bytes,
                alignment=dynamic.alignment,
                kind=RuntimePlacementKind.DYNAMIC_ACTION_DESTINATION,
            )
        )
    return tuple(result)


def _task_allocation_identities(
    layout: FixedPhysicalLayout,
) -> dict[tuple[str, int], tuple[int, ...]]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for task_id, ordinal, lease_id in layout.task_allocation_leases:
        grouped.setdefault((task_id, lease_id), []).append(ordinal)
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


def _action_identities(
    program: ShadowSpillProgram,
    schedule: MemorySchedule,
) -> dict[int, _ActionIdentity]:
    sizes = {item.alias_group_id: item.size_bytes for item in program.alias_groups}
    ordinals: dict[str, int] = {}
    result: dict[int, _ActionIdentity] = {}
    for index, action in enumerate(schedule.actions):
        if sizes[action.alias_group_id] == 0:
            continue
        ordinal = ordinals.get(action.trigger_task_id, 0)
        ordinals[action.trigger_task_id] = ordinal + 1
        result[index] = _ActionIdentity(
            task_id=canonical_index(action.trigger_task_id, "task_"),
            ordinal=ordinal,
            object_id=canonical_index(action.alias_group_id, "alias_"),
        )
        if action.kind is MemoryActionKind.RELEASE:
            continue
    return result


def _fixed_by_lease(placements: FixedLayoutPlacements) -> dict[int, _Fixed]:
    """Each fixed lease's offset, bytes and alignment, by lease id."""

    return dict(
        zip(
            placements.lease_id.tolist(),
            zip(
                placements.offset.tolist(),
                placements.bytes.tolist(),
                placements.alignment.tolist(),
                strict=True,
            ),
            strict=True,
        )
    )


def _fixed_runtime_placement(
    placement: _Fixed,
    *,
    task_id: int,
    ordinal: int,
    object_id: int,
    kind: RuntimePlacementKind,
) -> RuntimeFixedPlacement:
    offset, size, alignment = placement
    return RuntimeFixedPlacement(
        task_id=task_id,
        ordinal=ordinal,
        object_id=object_id,
        offset=offset,
        bytes=size,
        alignment=alignment,
        kind=kind,
    )


__all__ = ["DynamicTaskAllocationPolicy", "project_runtime_fixed_layout"]
