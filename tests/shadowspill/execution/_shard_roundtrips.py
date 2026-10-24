"""Exercise real descriptor transfers; no compiled Plan admission is claimed."""

import torch

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MutationSpec,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)
from shadowspill.pytorch.runtime_adapter.boundaries import (
    after_task_and_update,
    before_task_and_acquire,
    submit_initial_actions,
)
from shadowspill.runtime.plan import admit_initial_actions, admit_task, clear_tasks


def roundtrips(state, groups=None):
    bridge = state.bridge
    alias, value = "alias_000001", "object_000001"
    fetch = (MemoryAction("task_000000", alias, MemoryActionKind.FETCH),)
    evict = (MemoryAction("task_000001", alias, MemoryActionKind.EVICT),)
    task = TaskSpec(
        "task_000001",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "update",
        inputs=(value,),
        mutations=(MutationSpec(value),),
    )
    admit_initial_actions(bridge, fetch, task_number=0)
    handle = admit_task(bridge, task, (alias,), evict, trace_label="shard-update")
    region = state.regions[value]
    # Inspection must not introduce pageable copies or CPU waits inside a task.
    observed = torch.empty(15, dtype=torch.bfloat16, pin_memory=True)
    try:
        for step in range(3):
            submit_initial_actions(bridge, fetch, task_number=0)
            owner = state.owner(alias)
            before_task_and_acquire(bridge, handle, state.device.index, (owner,))
            weights = state.argument(value, state.geometry[value])
            if groups is not None:
                # The native fetch and this explicit task use the same ordered
                # NCCL group. Boundary submission must precede this call on every
                # worker, even though completion remains asynchronous.
                from shadowspill.communication.requests import CollectiveRequest
                from shadowspill.communication.task import PackedCollective

                members = next(
                    group
                    for group in state.plan.program.bindings["weight_shards"][alias][
                        "groups"
                    ]
                    if groups.logical_rank in group
                )
                PackedCollective(
                    CollectiveRequest(
                        "all_reduce", members, weights.numel() * 2, "torch.bfloat16"
                    ),
                    groups.for_members(members),
                )(weights)
                weights.div_(len(members))
            expected = torch.arange(1, 16, dtype=torch.bfloat16) + 2 * step
            observed.copy_(weights, non_blocking=True)
            weights.add_(2)
            del weights
            after_task_and_update(bridge, handle, state.device.index, (), (), (owner,))
            bridge.wait_until_idle()
            torch.testing.assert_close(observed, expected)
            host = state.host(alias).view(torch.bfloat16)
            expected_shard = torch.zeros(region.capacity, dtype=torch.bfloat16)
            expected_shard[: region.length] = (
                torch.arange(region.offset, region.offset + region.length)
                + 1
                + 2 * (step + 1)
            )
            torch.testing.assert_close(host, expected_shard)
            del host
    finally:
        clear_tasks(bridge)
