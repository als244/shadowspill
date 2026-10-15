"""A runtime trace is prepared with room for every record its invocation makes."""

from __future__ import annotations

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MutationSpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)
from shadowspill.runtime.plan import (
    TaskMemoryEnvelope,
    TraceCapacities,
    trace_capacities,
)
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
)

_COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


def _contract(allocations: int) -> TaskAllocationContract:
    return TaskAllocationContract.capture(
        [
            TaskAllocationEvent(ordinal, operation, 256, 256)
            for operation in (
                TaskAllocationOperation.ALLOCATE,
                TaskAllocationOperation.FREE,
            )
            for ordinal in range(allocations)
        ]
    )


def test_each_kind_of_record_is_counted_where_it_can_occur() -> None:
    control = TaskSpec(
        "task_000000", _COMPUTE, "profile_000000", requires_entrypoint=False
    )
    task = TaskSpec(
        "task_000001",
        _COMPUTE,
        "profile_000001",
        inputs=("object_a", "object_b", "object_c"),
        mutations=(MutationSpec("object_b"),),
    )
    actions = tuple(
        MemoryAction("task_000000", f"alias_{index:06d}", kind)
        for index, kind in enumerate(
            (
                MemoryActionKind.FETCH,
                MemoryActionKind.EVICT,
                MemoryActionKind.RELEASE,
                MemoryActionKind.WRITE_BACK,
            )
        )
    )

    capacities = trace_capacities(
        (
            (control, TaskMemoryEnvelope()),
            (task, TaskMemoryEnvelope(allocation_contract=_contract(2))),
        ),
        actions,
        floor=0,
    )

    session = 3
    tasks = 2 + (2 + 3 + 3 * 2)  # entered and left; three objects; two allocations
    copies = 3 * (1 + 3)  # fetch, evict, write-back: queued, then three records
    retirements = 2  # the evict and the release free a lease each
    release = 1  # queued
    assert capacities.events == session + tasks + copies + retirements + release
    assert capacities.allocation_events == 4 * (2 + len(actions))


def test_no_capacity_falls_below_the_floor() -> None:
    assert trace_capacities((), (), floor=64) == TraceCapacities(64, 64)
