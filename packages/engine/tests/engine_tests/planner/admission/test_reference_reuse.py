from __future__ import annotations

import ctypes
import random
from dataclasses import replace

import pytest

from reference.python.admission.reuse import (
    LayoutRefused,
    Operations,
    PlacedLease,
    Reuse,
    recover_layout_reuse,
)
from shadowspill.planner.capi import (
    CAdmissionOperations,
    CIndexedSchedule,
    CLayoutReuseProblem,
    CLayoutReuseResult,
    CLeaseIdentity,
    CLeaseLifetime,
    planner_api,
)
from shadowspill.status import ABI_VERSION

NO_INDEX = (1 << 32) - 1
NO_ID = (1 << 64) - 1
INVALID_ARGUMENT = 1
INVALID_OPERATIONS = 81


def _recover_compiled(
    leases: list[PlacedLease], operations: Operations, capacity: int | None = None
) -> tuple[Reuse, ...] | LayoutRefused | int:
    """The library's answer on the same columns the reference reads: the
    records, the refusal, or the count a too-small result was refused with."""

    count = len(leases)
    lifetimes = (CLeaseLifetime * max(count, 1))()
    identities = (CLeaseIdentity * max(count, 1))()
    offsets = (ctypes.c_uint64 * max(count, 1))()
    for index, lease in enumerate(leases):
        lifetimes[index].bytes = lease.bytes
        lifetimes[index].alignment = 1
        identities[index].lease_id = lease.lease_id
        identities[index].causal_start = lease.causal_start
        identities[index].causal_end = lease.causal_end
        identities[index].task = NO_INDEX if lease.task is None else lease.task
        identities[index].action = NO_INDEX if lease.action is None else lease.action
        offsets[index] = lease.offset
    operation_count = len(operations.boundaries)
    lease_count = len(operations.lease_retires)
    dependency_ids = (ctypes.c_uint64 * max(operation_count, 1))(
        *(NO_ID if item is None else item for item in operations.dependency_ids)
    )
    purposes = (ctypes.c_uint8 * max(operation_count, 1))(*operations.purposes)
    boundaries = (ctypes.c_uint8 * max(operation_count, 1))(*operations.boundaries)
    indices = (ctypes.c_uint32 * max(operation_count, 1))(*operations.indices)
    retires = (ctypes.c_uint64 * max(lease_count, 1))(
        *(NO_ID if item is None else item for item in operations.lease_retires)
    )
    compiled_operations = CAdmissionOperations(
        dependency_ids=dependency_ids,
        purposes=purposes,
        boundaries=boundaries,
        indices=indices,
        lease_retires=retires,
        operation_count=operation_count,
        lease_count=lease_count,
    )
    triggers = (ctypes.c_uint32 * max(len(operations.action_trigger_tasks), 1))(
        *operations.action_trigger_tasks
    )
    schedule = CIndexedSchedule(
        action_count=len(operations.action_trigger_tasks), action_trigger_tasks=triggers
    )
    problem = CLayoutReuseProblem(
        abi_version=ABI_VERSION,
        operations=ctypes.pointer(compiled_operations),
        schedule=ctypes.pointer(schedule),
        lifetimes=lifetimes,
        identities=identities,
        offsets=offsets,
        lifetime_count=count,
    )
    room = 4 * count + 8 if capacity is None else capacity
    columns = {
        "dependency_ids": (ctypes.c_uint64 * max(room, 1))(),
        "predecessor_leases": (ctypes.c_uint64 * max(room, 1))(),
        "predecessor_purposes": (ctypes.c_uint8 * max(room, 1))(),
        "predecessor_tasks": (ctypes.c_uint32 * max(room, 1))(),
        "predecessor_actions": (ctypes.c_uint32 * max(room, 1))(),
        "successor_leases": (ctypes.c_uint64 * max(room, 1))(),
        "successor_tasks": (ctypes.c_uint32 * max(room, 1))(),
        "successor_actions": (ctypes.c_uint32 * max(room, 1))(),
    }
    result = CLayoutReuseResult(capacity=room, **columns)
    status = planner_api().shadowspill_recover_layout_reuse(
        ctypes.byref(problem), ctypes.byref(result)
    )
    if status == INVALID_OPERATIONS:
        return LayoutRefused(
            result.refusal,
            result.refusal_predecessor,
            result.refusal_successor,
            "refused",
        )
    if status == INVALID_ARGUMENT:
        return int(result.count)
    assert status == 0
    return tuple(
        Reuse(
            dependency_id=result.dependency_ids[index],
            predecessor_lease_id=result.predecessor_leases[index],
            predecessor_purpose=result.predecessor_purposes[index],
            predecessor_task=result.predecessor_tasks[index],
            predecessor_action=(
                None
                if result.predecessor_actions[index] == NO_INDEX
                else result.predecessor_actions[index]
            ),
            successor_lease_id=result.successor_leases[index],
            successor_task=(
                None
                if result.successor_tasks[index] == NO_INDEX
                else result.successor_tasks[index]
            ),
            successor_action=(
                None
                if result.successor_actions[index] == NO_INDEX
                else result.successor_actions[index]
            ),
        )
        for index in range(result.count)
    )


def _layout(
    seed: int, count: int, slots: int = 6
) -> tuple[list[PlacedLease], Operations]:
    """A valid layout: leases take addresses in slots, later ones over earlier
    ones, every lease retired at an operation with a dependency before the
    next is acquired; some leases span two slots so one successor has several
    predecessors, and some sit on a boundary so adjacency is covered too."""

    rng = random.Random(seed)
    slot_bytes = [rng.choice((64, 128, 256)) for _ in range(slots)]
    slot_offsets = [sum(slot_bytes[:index]) for index in range(slots)]
    tasks = max(2, count // 2)
    actions = max(1, count // 3)
    leases: list[PlacedLease] = []
    retires: list[int | None] = []
    dependency_ids: list[int | None] = []
    purposes: list[int] = []
    boundaries: list[int] = []
    indices: list[int] = []
    for lease_id in range(count):
        slot = rng.randrange(slots)
        if rng.random() < 0.3 and slot + 1 < slots:
            offset = slot_offsets[slot] + rng.randrange(0, slot_bytes[slot], 16)
            end = slot_offsets[slot + 1] + rng.randrange(
                16, slot_bytes[slot + 1] + 1, 16
            )
        else:
            offset = slot_offsets[slot] + rng.randrange(0, slot_bytes[slot], 16)
            end = offset + rng.randrange(
                16, slot_offsets[slot] + slot_bytes[slot] - offset + 1, 16
            )
        boundary = rng.choice((1, 2, 3, 4))
        position = (
            rng.randrange(actions) if boundary in (3, 4) else rng.randrange(tasks)
        )
        task: int | None
        action: int | None
        kind = rng.random()
        if kind < 0.5:
            task, action = rng.randrange(tasks), None
        elif kind < 0.8:
            task, action = None, rng.randrange(actions)
        else:
            task, action = rng.randrange(tasks), rng.randrange(actions)
        leases.append(
            PlacedLease(
                lease_id=lease_id,
                offset=offset,
                bytes=end - offset,
                causal_start=lease_id * 10,
                causal_end=lease_id * 10 + 5,
                task=task,
                action=action,
            )
        )
        retires.append(lease_id)
        dependency_ids.append(rng.randrange(1_000, 9_999))
        purposes.append(rng.randrange(8))
        boundaries.append(boundary)
        indices.append(position)
    operations = Operations(
        dependency_ids=dependency_ids,
        purposes=purposes,
        boundaries=boundaries,
        indices=indices,
        lease_retires=retires,
        action_trigger_tasks=[rng.randrange(tasks) for _ in range(actions)],
    )
    return leases, operations


def _with(operations: Operations, **changes) -> Operations:
    fields = {name: list(getattr(operations, name)) for name in Operations.__slots__}
    for name, (index, value) in changes.items():
        fields[name][index] = value
    return Operations(**fields)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("count", (1, 2, 7, 40, 300))
def test_compiled_recovery_matches_readable_reference(seed: int, count: int) -> None:
    leases, operations = _layout(seed, count)
    expected = recover_layout_reuse(leases, operations)
    assert _recover_compiled(leases, operations) == expected
    if count >= 7:
        assert expected, "a layout this size shares addresses"


def test_recovery_is_independent_of_lease_listing_order() -> None:
    leases, operations = _layout(5, 60)
    shuffled = list(leases)
    random.Random(1).shuffle(shuffled)
    assert _recover_compiled(shuffled, operations) == _recover_compiled(
        leases, operations
    )


def test_a_result_too_small_is_refused_with_the_count_needed() -> None:
    leases, operations = _layout(3, 40)
    expected = recover_layout_reuse(leases, operations)
    assert _recover_compiled(leases, operations, capacity=1) == len(expected)
    assert _recover_compiled(leases, operations, capacity=len(expected)) == expected


def test_a_live_predecessor_refuses_the_layout() -> None:
    leases, operations = _layout(7, 12)
    first = min(leases, key=lambda item: item.causal_start)
    second = next(
        item
        for item in sorted(leases, key=lambda item: item.causal_start)
        if item is not first
        and item.offset < first.offset + first.bytes
        and first.offset < item.offset + item.bytes
    )
    leases = [
        replace(item, causal_end=second.causal_start + 1) if item is first else item
        for item in leases
    ]
    with pytest.raises(LayoutRefused) as refused:
        recover_layout_reuse(leases, operations)
    compiled = _recover_compiled(leases, operations)
    assert isinstance(compiled, LayoutRefused)
    assert (
        (compiled.kind, compiled.predecessor, compiled.successor)
        == (
            refused.value.kind,
            refused.value.predecessor,
            refused.value.successor,
        )
        == (1, first.lease_id, second.lease_id)
    )


@pytest.mark.parametrize(
    "change, kind",
    (
        ({"dependency_ids": (0, None)}, 2),
        ({"lease_retires": (0, None)}, 2),
        ({"boundaries": (0, 0)}, 3),
    ),
)
def test_a_predecessor_without_a_completion_refuses_the_layout(change, kind) -> None:
    leases, operations = _layout(9, 12)
    # Make lease 0 the one every later overlapping lease reuses first.
    leases[0] = replace(
        leases[0], offset=0, bytes=sum(item.offset + item.bytes for item in leases)
    )
    operations = _with(operations, **change)
    with pytest.raises(LayoutRefused) as refused:
        recover_layout_reuse(leases, operations)
    compiled = _recover_compiled(leases, operations)
    assert isinstance(compiled, LayoutRefused)
    assert compiled.kind == refused.value.kind == kind
    assert compiled.predecessor == refused.value.predecessor == 0


def test_a_successor_without_a_consumer_refuses_the_layout() -> None:
    leases, operations = _layout(4, 12)
    leases[0] = replace(leases[0], offset=0, bytes=4096)
    leases[1] = replace(leases[1], offset=0, bytes=64, task=None, action=None)
    with pytest.raises(LayoutRefused) as refused:
        recover_layout_reuse(leases, operations)
    compiled = _recover_compiled(leases, operations)
    assert isinstance(compiled, LayoutRefused)
    assert compiled.kind == refused.value.kind == 4
    assert compiled.successor == refused.value.successor == 1


def test_an_empty_layout_has_no_dependencies() -> None:
    assert _recover_compiled([], Operations([], [], [], [], [], [])) == ()
