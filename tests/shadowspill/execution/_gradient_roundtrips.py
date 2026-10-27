"""Native RS/replica-AR eviction and zero/owned-shard restore across calls."""

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
from shadowspill.runtime.plan import (
    abort_task,
    admit_initial_actions,
    admit_task,
    clear_tasks,
)


def roundtrips(state, groups):
    bridge = state.bridge
    alias, value = "alias_000001", "object_000001"
    fetch = (MemoryAction("task_000000", alias, MemoryActionKind.FETCH),)
    evict = (MemoryAction("task_000001", alias, MemoryActionKind.EVICT),)
    task = TaskSpec(
        "task_000001",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "contribute",
        inputs=(value,),
        mutations=(MutationSpec(value),),
    )
    admit_initial_actions(bridge, fetch, task_number=0)
    handle = admit_task(bridge, task, (alias,), evict, trace_label="gradient-add")
    region = state.regions[value]
    dtype = getattr(torch, region.geometry["dtype"].removeprefix("torch."))
    description = state.plan.program.bindings["gradient_shards"][alias]
    replicas = next(
        (g for g in description["replica_groups"] if groups.logical_rank in g),
        (groups.logical_rank,),
    )
    expected = torch.arange(1, 17, dtype=dtype)
    expected[15] = 0
    observed = torch.empty(15, dtype=dtype, pin_memory=True)
    world = state.plan.program.parallelism.world_size
    active = False
    try:
        for step in range(3):
            before = torch.zeros(15, dtype=dtype)
            end = region.offset + region.length
            if replicas.index(groups.logical_rank) == 0:
                before[region.offset : end] = expected[region.offset : end]
            submit_initial_actions(bridge, fetch, task_number=0)
            owner = state.owner(alias)
            before_task_and_acquire(bridge, handle, state.device.index, (owner,))
            active = True
            gradient = state.argument(value, state.geometry[value])
            observed.copy_(gradient, non_blocking=True)
            # Distinct rank-local contributions, two microbatches per spill.
            for contribution in (1, 2):
                gradient.add_((groups.logical_rank + 1) * (step + 1) * contribution)
                expected[:15].add_(sum(range(1, world + 1)) * (step + 1) * contribution)
            del gradient
            after_task_and_update(bridge, handle, state.device.index, (), (), (owner,))
            active = False
            bridge.wait_until_idle()
            torch.testing.assert_close(observed, before, rtol=0, atol=0)
            padded = torch.zeros(region.capacity, dtype=dtype)
            padded[: region.length] = expected[region.offset : end]
            with state.spill.buffer(alias, 0, state.spill.sizes[alias]) as host:
                torch.testing.assert_close(host.view(dtype), padded, rtol=0, atol=0)
            del host
    finally:
        if active:
            abort_task(bridge, handle)
        clear_tasks(bridge)
