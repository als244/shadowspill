"""Admitted invocation paths share persistent tensors and a single fixed arena."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.plan import Plan
from shadowspill.planning.shared import shared_memory
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
    shared_aliases: Mapping[str, str] | None = None,
    extra_initial: tuple[str, ...] = (),
    shared_slice_bytes: int | None = None,
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
        )
        verify_initialization(storage)
        controls = Controls(storage)
        tasks, acquisition = prepare(
            plan, bridge, shared_slice_bytes=shared_slice_bytes
        )
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
    plans: Mapping[str, Plan] | None = None,
) -> tuple[Runtime, tuple[ExecutionPath, ...], dict[str, ExecutionPath]]:
    plans = dict(plans or {})
    joint = shared_memory(plan, plans) if plans else None
    runtime, memory, bridge = open_worker(plan, device, external_headroom_bytes)
    paths = []
    named = {}
    try:
        other = plan._accumulation_plan
        extra = (
            ()
            if other is None
            else tuple(r.alias_group_id for r in other.schedule.initial_residency)
        )
        if joint is not None:
            extra = tuple(
                set(extra)
                | {
                    alias
                    for mapping in joint.aliases.values()
                    for alias in mapping.values()
                }
            )
        paths.append(
            _bind(
                runtime,
                plan,
                memory,
                bridge,
                extra_initial=extra,
                shared_slice_bytes=None if joint is None else joint.fixed_slice_bytes,
            )
        )
        owner = memory.plan_handle
        if other is not None:
            memory, bridge = open_path(runtime, other, device, slab_owner=owner)
            paths.append(_bind(runtime, other, memory, bridge, shared=paths[0].storage))
        for name, additional in plans.items():
            assert joint is not None
            memory, bridge = open_path(runtime, additional, device, slab_owner=owner)
            named[name] = _bind(
                runtime,
                additional,
                memory,
                bridge,
                shared=paths[0].storage,
                shared_aliases=joint.aliases[name],
            )
        return runtime, tuple(paths), named
    except BaseException:
        for path in reversed([*paths, *named.values()]):
            path.close(runtime)
        runtime.close()
        raise
