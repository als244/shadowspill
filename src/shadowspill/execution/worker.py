"""Admitted invocation paths share persistent tensors and a single fixed arena."""

from dataclasses import dataclass

import torch

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
from .setup import open_path, open_worker
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


def _bind(
    runtime: Runtime,
    plan: Plan,
    memory: PlanMemory,
    bridge: RuntimeBridge,
    *,
    shared: Storage | None = None,
    extra_initial: tuple[str, ...] = (),
) -> ExecutionPath:
    try:
        storage = Storage(
            plan,
            bridge,
            torch.device("cuda", memory.execution_device),
            shared=shared,
            extra_initial=extra_initial,
        )
        verify_initialization(storage)
        controls = Controls(storage)
        tasks, acquisition = prepare(plan, bridge)
        adopt_plan(runtime, memory.plan_handle)
        return ExecutionPath(
            plan, memory, bridge, storage, controls, tasks, acquisition
        )
    except BaseException:
        bridge.objects.unregister(bridge.objects.registered_aliases())
        abort_plan(runtime, memory.plan_handle)
        raise


def create_paths(
    plan: Plan,
    device: int,
    external_headroom_bytes: int,
) -> tuple[Runtime, tuple[ExecutionPath, ...]]:
    runtime, memory, bridge = open_worker(plan, device, external_headroom_bytes)
    paths = []
    try:
        other = plan._accumulation_plan
        extra = (
            ()
            if other is None
            else tuple(r.alias_group_id for r in other.schedule.initial_residency)
        )
        paths.append(_bind(runtime, plan, memory, bridge, extra_initial=extra))
        if other is not None:
            memory, bridge = open_path(
                runtime, other, device, slab_owner=memory.plan_handle
            )
            paths.append(_bind(runtime, other, memory, bridge, shared=paths[0].storage))
        return runtime, tuple(paths)
    except BaseException:
        for path in reversed(paths):
            path.close(runtime)
        runtime.close()
        raise
