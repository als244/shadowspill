"""Bind and release one admitted invocation path in a worker."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.plan import Plan
from shadowspill.runtime import Runtime
from shadowspill.runtime.plan import RuntimeBridge, clear_tasks
from shadowspill.runtime.plan.lifecycle import (
    PlanMemory,
    abort_plan,
    adopt_plan,
    release_plan,
)

from .admission import BoundTask, prepare
from .controls import Controls
from .distributed.worker import DistributedWorker
from .storage import Storage
from .tensors import verify_initialization


@dataclass
class ExecutionPath:
    plan: Plan
    memory: PlanMemory
    bridge: RuntimeBridge
    storage: Storage
    controls: Controls
    tasks: tuple[BoundTask, ...]
    acquisition: int

    def close(self, runtime: Runtime) -> None:
        self.storage.clear_views()
        clear_tasks(self.bridge)
        self.bridge.objects.unregister(self.bridge.objects.registered_aliases())
        release_plan(runtime, self.memory.plan_handle)


def bind_path(
    runtime: Runtime,
    plan: Plan,
    memory: PlanMemory,
    bridge: RuntimeBridge,
    *,
    shared: Storage | None = None,
    shared_aliases: Mapping[str, str] | None = None,
    extra_initial: tuple[str, ...] = (),
    shared_slice_bytes: int | None = None,
    distributed: DistributedWorker | None = None,
) -> ExecutionPath:
    try:
        require_compiled_device(plan.program, memory.execution_device)
        storage = Storage(
            plan,
            bridge,
            torch.device("cuda", memory.execution_device),
            shared=shared,
            shared_aliases=shared_aliases,
            extra_initial=extra_initial,
            logical_rank=None
            if distributed is None
            else distributed.groups.logical_rank,
            transfer_descriptors=None
            if distributed is None
            else distributed.descriptors(plan),
        )
        verify_initialization(storage)
        controls = Controls(storage)
        tasks, acquisition = prepare(
            plan,
            bridge,
            shared_slice_bytes=shared_slice_bytes,
            groups=None if distributed is None else distributed.groups,
        )
        adopt_plan(runtime, memory.plan_handle)
        return ExecutionPath(
            plan, memory, bridge, storage, controls, tasks, acquisition
        )
    except BaseException:
        bridge.objects.unregister(bridge.objects.registered_aliases())
        abort_plan(runtime, memory.plan_handle)
        raise
