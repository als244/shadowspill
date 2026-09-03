"""Project a semantic physical certificate into indexed runtime identities."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.ir import MemoryActionKind, MemorySchedule, Program
from shadowspill.ir.schedule import first_use_initial_order
from shadowspill.planner.admission.admission_replay import AdmissionReplayPurpose
from shadowspill.planner.admission.layout.model import (
    FixedLayoutPlacement,
    FixedPhysicalLayout,
    LeaseLifetime,
)
from shadowspill.pytorch.runtime_adapter.fixed_layout import (
    RuntimeFixedDependency,
    RuntimeFixedLayout,
    RuntimeFixedPlacement,
    RuntimePlacementKind,
)

_NO_ID = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class _ActionIdentity:
    task_id: int
    ordinal: int
    object_id: int


@dataclass(frozen=True, slots=True)
class DynamicTaskAllocationPolicy:
    """One task allocation served outside the reusable fixed-layout slice."""

    task_id: str
    allocation_ordinal: int
    bytes: int
    alignment: int


def project_runtime_fixed_layout(
    layout: FixedPhysicalLayout,
    program: Program,
    schedule: MemorySchedule,
    *,
    initial_task_id: int,
    dynamic_task_allocations: tuple[DynamicTaskAllocationPolicy, ...] = (),
) -> RuntimeFixedLayout:
    """Translate one layout without changing its placement or dependency policy."""

    if layout.program_digest != program.digest:
        raise ValueError("fixed layout belongs to a different Program")
    if layout.schedule_digest != schedule.digest:
        raise ValueError("fixed layout belongs to a different memory schedule")
    if initial_task_id == _NO_ID:
        raise ValueError("initial-placement task cannot use the no-ID sentinel")

    fixed_by_lease = {item.lease_id: item for item in layout.placements}
    dynamic_by_lease = {item.lease_id: item for item in layout.dynamic_lifetimes}
    action_identities = _action_identities(program, schedule)
    task_allocations = _task_allocation_identities(layout)

    placements = [
        *_initial_placements(
            layout,
            program,
            schedule,
            fixed_by_lease,
            initial_task_id=initial_task_id,
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
    dependencies = _runtime_dependencies(
        layout,
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
        initial_task_id=initial_task_id,
    )


def _initial_placements(
    layout: FixedPhysicalLayout,
    program: Program,
    schedule: MemorySchedule,
    fixed_by_lease: dict[int, FixedLayoutPlacement],
    *,
    initial_task_id: int,
) -> tuple[RuntimeFixedPlacement, ...]:
    initial_leases = dict(layout.initial_alias_leases)
    sizes = {item.alias_group_id: item.size_bytes for item in program.alias_groups}
    # The batch pairs action N with ACTION_DESTINATION ordinal N, so the
    # ordinals here must follow the same first-use order the executors
    # submit the fetches in.
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
            task_id=initial_task_id,
            ordinal=ordinal,
            object_id=_plan_index(alias_id, "alias_"),
            kind=RuntimePlacementKind.ACTION_DESTINATION,
        )
        for ordinal, alias_id in enumerate(aliases)
    )


def _task_placements(
    layout: FixedPhysicalLayout,
    fixed_by_lease: dict[int, FixedLayoutPlacement],
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
                    task_id=_plan_index(task_id, "task_"),
                    ordinal=ordinal,
                    object_id=_NO_ID,
                    kind=RuntimePlacementKind.TASK_ALLOCATION,
                )
            )
            continue
        dynamic = dynamic_by_lease.get(lease_id)
        if dynamic is None:
            raise ValueError(f"task allocation lease {lease_id} has no policy")
        result.append(
            RuntimeFixedPlacement(
                task_id=_plan_index(task_id, "task_"),
                ordinal=ordinal,
                object_id=_NO_ID,
                offset=_NO_ID,
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
                task_id=_plan_index(policy.task_id, "task_"),
                ordinal=policy.allocation_ordinal,
                object_id=_NO_ID,
                offset=_NO_ID,
                bytes=policy.bytes,
                alignment=policy.alignment,
                kind=RuntimePlacementKind.DYNAMIC_TASK_ALLOCATION,
            )
        )
    return tuple(result)


def _action_placements(
    layout: FixedPhysicalLayout,
    fixed_by_lease: dict[int, FixedLayoutPlacement],
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
                offset=_NO_ID,
                bytes=dynamic.bytes,
                alignment=dynamic.alignment,
                kind=RuntimePlacementKind.DYNAMIC_ACTION_DESTINATION,
            )
        )
    return tuple(result)


def _runtime_dependencies(
    layout: FixedPhysicalLayout,
    action_identities: dict[int, _ActionIdentity],
    task_allocations: dict[tuple[str, int], tuple[int, ...]],
) -> tuple[RuntimeFixedDependency, ...]:
    result: set[RuntimeFixedDependency] = set()
    for item in layout.reuse_dependencies:
        if item.predecessor_purpose is not AdmissionReplayPurpose.EVICTION:
            continue
        if item.predecessor_action_index is None:
            raise ValueError("fixed eviction dependency has no action identity")
        predecessor = action_identities[item.predecessor_action_index]
        if item.successor_action_index is not None:
            successor = action_identities[item.successor_action_index]
            result.add(
                RuntimeFixedDependency(
                    predecessor_task_id=predecessor.task_id,
                    predecessor_action_ordinal=predecessor.ordinal,
                    successor_task_id=successor.task_id,
                    successor_ordinal=successor.ordinal,
                    successor_kind=RuntimePlacementKind.ACTION_DESTINATION,
                )
            )
            continue
        if item.successor_task_id is None:
            raise ValueError("fixed reuse dependency has no successor")
        ordinals = task_allocations.get(
            (item.successor_task_id, item.successor_lease_id)
        )
        if not ordinals:
            raise ValueError(
                f"fixed task lease {item.successor_lease_id} has no ordinal"
            )
        result.add(
            RuntimeFixedDependency(
                predecessor_task_id=predecessor.task_id,
                predecessor_action_ordinal=predecessor.ordinal,
                successor_task_id=_plan_index(item.successor_task_id, "task_"),
                successor_ordinal=min(ordinals),
                successor_kind=RuntimePlacementKind.TASK_ALLOCATION,
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
    program: Program,
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
            task_id=_plan_index(action.trigger_task_id, "task_"),
            ordinal=ordinal,
            object_id=_plan_index(action.alias_group_id, "alias_"),
        )
        if action.kind is MemoryActionKind.RELEASE:
            continue
    return result


def _fixed_runtime_placement(
    placement: FixedLayoutPlacement,
    *,
    task_id: int,
    ordinal: int,
    object_id: int,
    kind: RuntimePlacementKind,
) -> RuntimeFixedPlacement:
    return RuntimeFixedPlacement(
        task_id=task_id,
        ordinal=ordinal,
        object_id=object_id,
        offset=placement.offset,
        bytes=placement.bytes,
        alignment=placement.alignment,
        kind=kind,
    )


def _plan_index(value: str, prefix: str) -> int:
    if not value.startswith(prefix):
        raise ValueError(f"runtime identity {value!r} lacks prefix {prefix!r}")
    suffix = value.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"runtime identity {value!r} has a nonnumeric suffix")
    return int(suffix)


__all__ = ["DynamicTaskAllocationPolicy", "project_runtime_fixed_layout"]
