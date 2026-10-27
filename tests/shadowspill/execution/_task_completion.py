"""Fresh allocator worker: two streams repeatedly reuse the same fixed address."""

import sys

import torch

from shadowspill.ir import (
    DeviceSpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskSpec,
)
from shadowspill.memory import DevicePool, PinnedHostPool, TransferRoute
from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.pytorch.runtime_adapter.boundaries import (
    after_task_and_update,
    before_task_and_acquire,
    wait_task_allocations,
)
from shadowspill.runtime import Runtime
from shadowspill.runtime.admission_capi import NO_ID
from shadowspill.runtime.fixed_layout import (
    RuntimeFixedDependency,
    RuntimeFixedLayout,
    RuntimeFixedPlacement,
    RuntimePlacementKind,
)
from shadowspill.runtime.plan.admission import (
    admit_fixed_layout,
    admit_task,
    seal_fixed_layout,
)
from shadowspill.runtime.plan.bridge import RuntimeBridge
from shadowspill.runtime.plan.common import TaskMemoryEnvelope
from shadowspill.runtime.plan.lifecycle import abort_plan, begin_plan
from shadowspill.task.allocations.contract import TaskAllocationContract
from shadowspill.task.allocations.events import (
    TaskAllocationEvent,
    TaskAllocationOperation,
)


def new_plan(runtime):
    memory = begin_plan(
        runtime,
        execution="execution",
        spill="spill",
        execution_device=0,
    )
    ir = ShadowSpillProgram(
        devices=(DeviceSpec("cuda_0", "worker", "cuda", 0),),
        alias_groups=(),
        objects=(),
        tasks=(),
        profiles=(),
    )
    bridge = RuntimeBridge(
        runtime,
        ir,
        memory.plan_handle,
        execution_pool_id=memory.execution.pool_id,
        spill_pool_id=memory.spill.pool_id,
    )
    return memory, bridge


def exercise(runtime, kind):
    memory, bridge = new_plan(runtime)
    try:
        size = 16_384
        layout = RuntimeFixedLayout(
            size,
            tuple(
                RuntimeFixedPlacement(
                    task, 0, NO_ID, 0, size, 256, RuntimePlacementKind.TASK_ALLOCATION
                )
                for task in (1, 2)
            ),
            (
                RuntimeFixedDependency(
                    1,
                    NO_ID,
                    2,
                    NO_ID if kind == RuntimePlacementKind.TASK_START else 0,
                    kind,
                ),
            ),
        )
        admit_fixed_layout(bridge, layout)
        contract = TaskAllocationContract.capture(
            [
                TaskAllocationEvent(0, operation, size, size)
                for operation in (
                    TaskAllocationOperation.ALLOCATE,
                    TaskAllocationOperation.FREE,
                )
            ]
        )
        handles = [
            admit_task(
                bridge,
                TaskSpec(f"task_{index:06d}", ResourceSpec("cuda_0", resource), "test"),
                (),
                (),
                memory_envelope=TaskMemoryEnvelope(allocation_contract=contract),
                trace_label=f"completion/{index}",
            )
            for index, resource in (
                (1, ResourceKind.COMMUNICATION),
                (2, ResourceKind.COMPUTE),
            )
        ]
        seal_fixed_layout(bridge)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        origin = torch.cuda.Event(enable_timing=True)
        finished = [torch.cuda.Event(enable_timing=True) for _ in streams]
        observed = [torch.empty(size // 4, pin_memory=True) for _ in streams]
        for iteration in range(3):
            addresses = []
            origin.record(streams[0])
            streams[1].wait_event(origin)
            for index, (stream, handle) in enumerate(
                zip(streams, handles, strict=True)
            ):
                with torch.cuda.stream(stream):
                    before_task_and_acquire(handle, 0, ())
                    wait_task_allocations(handle, 0)
                    value = torch.empty(size // 4, device="cuda")
                    addresses.append(value.data_ptr())
                    if index == 0:
                        torch.cuda._sleep(50_000_000)
                    value.fill_(iteration * 2 + index)
                    observed[index].copy_(value, non_blocking=True)
                    del value
                    after_task_and_update(handle, 0, (), ())
                    finished[index].record()
            bridge.wait_until_idle()
            for stream in streams:
                stream.synchronize()
            assert addresses[0] == addresses[1]
            assert origin.elapsed_time(finished[1]) >= origin.elapsed_time(finished[0])
            for index, output in enumerate(observed):
                torch.testing.assert_close(
                    output, torch.full_like(output, iteration * 2 + index)
                )
    finally:
        abort_plan(runtime, memory.plan_handle)


def exercise_private_stream_reuse(runtime):
    """Free on the host while a side stream still needs the old slot contents."""
    memory, bridge = new_plan(runtime)
    size = 16_384
    try:
        admit_fixed_layout(
            bridge,
            RuntimeFixedLayout(
                size,
                tuple(
                    RuntimeFixedPlacement(
                        1,
                        ordinal,
                        NO_ID,
                        0,
                        size,
                        256,
                        RuntimePlacementKind.TASK_ALLOCATION,
                    )
                    for ordinal in (0, 1)
                ),
                (),
            ),
        )
        contract = TaskAllocationContract.capture(
            [
                TaskAllocationEvent(ordinal, operation, size, size)
                for ordinal in (0, 1)
                for operation in (
                    TaskAllocationOperation.ALLOCATE,
                    TaskAllocationOperation.FREE,
                )
            ]
        )
        handle = admit_task(
            bridge,
            TaskSpec(
                "task_000001", ResourceSpec("cuda_0", ResourceKind.COMPUTE), "test"
            ),
            (),
            (),
            memory_envelope=TaskMemoryEnvelope(allocation_contract=contract),
            trace_label="completion/private-stream-reuse",
        )
        seal_fixed_layout(bridge)
        side = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        observed = [torch.empty(size // 4, pin_memory=True) for _ in range(2)]
        for iteration in range(24):
            before_task_and_acquire(handle, 0, ())
            wait_task_allocations(handle, 0)
            first = torch.empty(size // 4, device="cuda")
            pointer = first.data_ptr()
            first.fill_(iteration * 2)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                torch.cuda._sleep(50_000_000 if iteration % 8 == 0 else 0)
                observed[0].copy_(first, non_blocking=True)
            first.record_stream(side)
            del first
            second = torch.empty(size // 4, device="cuda")
            assert second.data_ptr() == pointer
            second.fill_(iteration * 2 + 1)
            observed[1].copy_(second, non_blocking=True)
            del second
            current.wait_stream(side)
            after_task_and_update(handle, 0, (), ())
            bridge.wait_until_idle()
            torch.cuda.synchronize()
            for index, result in enumerate(observed):
                torch.testing.assert_close(
                    result, torch.full_like(result, iteration * 2 + index)
                )
    finally:
        abort_plan(runtime, memory.plan_handle)


assert not torch.cuda.is_initialized()
with Runtime(
    frontend=PyTorchFrontend(),
    calibrate=False,
    pools={
        "execution": DevicePool(2 << 30, external_headroom=256 << 20),
        "spill": PinnedHostPool(1 << 20),
    },
    routes={
        "fetch": TransferRoute("spill", "execution"),
        "evict": TransferRoute("execution", "spill"),
    },
) as runtime:
    if sys.argv[1] == "INTRA_TASK":
        exercise_private_stream_reuse(runtime)
    else:
        exercise(runtime, RuntimePlacementKind[sys.argv[1]])
print("task completion passed")
