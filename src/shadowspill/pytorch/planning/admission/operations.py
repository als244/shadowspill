"""Give a compiled operation sequence its identifiers back.

A lease's provenance is that of the operation which acquired it, except for
task allocations: a task may free a slot and reallocate it, and a reallocation
emits no operation of its own. Those steps are replayed alongside the sequence,
so a reused lease records what it most recently became rather than what it
first was. That is the role the fixed/dynamic split checks - a slot first used
as workspace and then retained as an output really is an output by the end of
the task - and it is why `_validate_dynamic_lifetimes` accepts it.

`shadowspill_build_admission_operations` returns one schedule's pool
operations as indexed columns. Everything below - lifetime construction,
placement, the layout certificate - works in identifiers, so this walks the
sequence once and produces the identified steps plus the four lease maps the
certificate needs.

The walk derives rather than decides: a lease's provenance is the provenance
of the operation that acquired it, and each map is the subset of acquisitions
with one purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.planner._operations import AdmissionOperations
from shadowspill.runtime import (
    AdmissionReplayOperation,
    AdmissionReplayOperationKind,
)

from .admission_replay import (
    AdmissionReplayPurpose,
    AdmissionReplayStep,
    _LeaseProvenance,
)


@dataclass(frozen=True, slots=True)
class AllocationStep:
    """One task allocation step, in the order the topology flattens them."""

    task_id: str
    ordinal: int
    slot: int
    allocates: bool
    purpose: AdmissionReplayPurpose
    alias_group_id: str | None

#: Compiled purpose codes, in the order `ShadowSpillAdmissionPurpose` declares.
_PURPOSES = (
    AdmissionReplayPurpose.INITIAL_OBJECT,
    AdmissionReplayPurpose.TASK_WORKSPACE,
    AdmissionReplayPurpose.TASK_OUTPUT,
    AdmissionReplayPurpose.MUTATION_REPLACEMENT,
    AdmissionReplayPurpose.RELEASE,
    AdmissionReplayPurpose.EVICTION,
    AdmissionReplayPurpose.FETCH_DESTINATION,
    AdmissionReplayPurpose.TERMINAL_COMPLETION,
)

#: Compiled boundary codes, in the order `ShadowSpillAdmissionBoundaryKind`
#: declares. Initial residency names neither a task nor an action, and its
#: index carries no meaning; the rest name one or the other.
_INITIAL_BOUNDARY = 0
_ACTION_BOUNDARIES = frozenset({3, 4})

_ACQUISITIONS = frozenset(
    {
        AdmissionReplayOperationKind.ACQUIRE,
        AdmissionReplayOperationKind.RESERVE,
    }
)

_RETIREMENTS = frozenset(
    {
        AdmissionReplayOperationKind.BEGIN_RETIREMENT,
        AdmissionReplayOperationKind.RELEASE,
    }
)


@dataclass(frozen=True, slots=True)
class IdentifiedOperations:
    """One schedule's operations, and the leases they created, by identifier."""

    steps: tuple[AdmissionReplayStep, ...]
    lease_provenance: dict[int, _LeaseProvenance]
    active_aliases: dict[str, int]
    initial_alias_leases: dict[str, int]
    task_allocation_leases: dict[tuple[str, int], int]
    action_destination_leases: dict[int, int]
    fetch_bytes: int
    evict_bytes: int


def identify_operations(
    operations: AdmissionOperations,
    *,
    task_ids: tuple[str, ...],
    alias_ids: tuple[str, ...],
    allocation_steps: tuple[AllocationStep, ...],
    action_trigger_tasks: tuple[int, ...],
    storage_handoffs: tuple[tuple[str, str], ...],
) -> IdentifiedOperations:
    """Walk the sequence once, resolving indices to identifiers.

    `allocation_steps` is indexed by flattened allocation offset, which is what
    an operation records when it acquires a task lease. Replaying those steps
    resolves the leases a reallocation reuses without inventing operations for
    them, and supplies the alias each step owns.

    `action_trigger_tasks` gives each action's triggering task, because an
    action-boundary operation belongs to both: the action names what happened
    and the task names when.

    `storage_handoffs` lists (source, destination) alias pairs in task order. A
    handoff moves a live lease between aliases without allocating, so it emits
    no operation and has to be replayed for the active-alias map to end
    correct.
    """

    steps: list[AdmissionReplayStep] = []
    provenance: dict[int, _LeaseProvenance] = {}
    active: dict[str, int] = {}
    initial: dict[str, int] = {}
    allocations: dict[tuple[str, int], int] = {}
    destinations: dict[int, int] = {}
    slot_leases: dict[int, int] = {}

    for sequence, kind_code in enumerate(operations.kinds):
        kind = AdmissionReplayOperationKind(kind_code)
        lease_id = operations.lease_ids[sequence]
        purpose = _PURPOSES[operations.purposes[sequence]]
        boundary = operations.boundaries[sequence]
        index = operations.indices[sequence]
        is_action = boundary in _ACTION_BOUNDARIES
        is_initial = boundary == _INITIAL_BOUNDARY
        action_index = index if is_action else None
        if is_initial:
            task_id = None
        elif is_action:
            task_id = task_ids[action_trigger_tasks[index]]
        else:
            task_id = task_ids[index]
        offset = operations.allocation_offsets[sequence]
        if offset is not None:
            # A task allocation owns whatever its step declares, which is
            # None for anonymous workspace.
            alias_id = allocation_steps[offset].alias_group_id
        else:
            alias_slot = (
                operations.lease_aliases[lease_id]
                if lease_id < len(operations.lease_aliases)
                else None
            )
            alias_id = None if alias_slot is None else alias_ids[alias_slot]

        step_provenance = _LeaseProvenance(
            purpose,
            task_id=task_id,
            alias_group_id=alias_id,
            action_index=action_index,
        )
        steps.append(
            AdmissionReplayStep(
                AdmissionReplayOperation(
                    sequence=sequence,
                    lease_id=lease_id,
                    kind=kind,
                    bytes=operations.bytes[sequence],
                    alignment=operations.alignments[sequence],
                    dependency_id=None,
                    dependency_expected=False,
                ),
                purpose,
                task_id=task_id,
                alias_group_id=alias_id,
                action_index=action_index,
            )
        )
        if (
            kind in _RETIREMENTS
            and alias_id is not None
            and active.get(alias_id) == lease_id
        ):
            del active[alias_id]
        if kind not in _ACQUISITIONS or lease_id in provenance:
            continue
        provenance[lease_id] = step_provenance
        if alias_id is not None:
            active[alias_id] = lease_id
        if purpose is AdmissionReplayPurpose.INITIAL_OBJECT and alias_id:
            initial[alias_id] = lease_id
        elif (
            purpose is AdmissionReplayPurpose.FETCH_DESTINATION
            and action_index is not None
        ):
            destinations[action_index] = lease_id
        else:
            if offset is not None:
                slot_leases[allocation_steps[offset].slot] = lease_id

    for step in allocation_steps:
        if not step.allocates:
            continue
        lease = slot_leases.get(step.slot)
        if lease is None:
            continue
        allocations[(step.task_id, step.ordinal)] = lease
        provenance[lease] = _LeaseProvenance(
            step.purpose,
            task_id=step.task_id,
            alias_group_id=step.alias_group_id,
        )

    for source, destination in storage_handoffs:
        lease = active.pop(source, None)
        if lease is not None:
            active[destination] = lease

    return IdentifiedOperations(
        steps=tuple(steps),
        lease_provenance=provenance,
        active_aliases=active,
        initial_alias_leases=initial,
        task_allocation_leases=allocations,
        action_destination_leases=destinations,
        fetch_bytes=operations.fetch_bytes,
        evict_bytes=operations.evict_bytes,
    )


__all__ = ["AllocationStep", "IdentifiedOperations", "identify_operations"]
