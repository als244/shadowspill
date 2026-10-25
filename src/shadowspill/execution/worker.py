"""Admitted invocation paths share persistent tensors and a single fixed arena."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.plan import Plan
from shadowspill.ir import MemoryActionKind
from shadowspill.pipeline.admission.physical import seal_worker_budget
from shadowspill.planning.shared import shared_memory
from shadowspill.pytorch.resource_streams import ResourceStreams
from shadowspill.runtime import Runtime
from shadowspill.runtime.bootstrap import installed_runtime
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


def create_paths(
    plan: Plan,
    device: int,
    external_headroom_bytes: int | None,
    plans: Mapping[str, Plan] | None = None,
    rank_map: Mapping[int, int] | None = None,
) -> tuple[
    Runtime,
    tuple[ExecutionPath, ...],
    dict[str, ExecutionPath],
    ResourceStreams,
    DistributedWorker | None,
]:
    plans = dict(plans or {})
    joint = shared_memory(plan, plans) if plans else None
    reservations = [
        p.profiled_program.metadata.get("external_headroom_bytes")
        for p in (plan, *plans.values())
    ]
    reserved = max(value if value is not None else 512 << 20 for value in reservations)
    if external_headroom_bytes is None:
        external_headroom_bytes = reserved
    elif (
        all(value is not None for value in reservations)
        and external_headroom_bytes > reserved
    ):
        raise ValueError(
            "Executor external_headroom_bytes exceeds the Plan reservation; "
            "replan with profiles that include the requested headroom"
        )
    distributed = None
    if plan.program.parallelism is not None and plan.program.parallelism.world_size > 1:
        all_plans = {
            "default": plan,
            **{f"named.{name}": p for name, p in plans.items()},
        }
        if plan._accumulation_plan is not None:
            all_plans["accumulate"] = plan._accumulation_plan
        distributed = DistributedWorker(all_plans, rank_map)
        calibrations = {
            digest(p.communication_info.to_dict()): p.communication_info
            for p in all_plans.values()
        }
        external_headroom_bytes += sum(
            info.fixed_device_bytes or 0 for info in calibrations.values()
        )
    try:
        runtime = open_worker(plan, device, external_headroom_bytes)
    except BaseException:
        if distributed is not None:
            distributed.close()
        raise
    paths = []
    named = {}
    try:
        if distributed is not None:
            distributed.prepare(device)
        selected = [
            (task, candidate)
            for candidate in (plan, plan._accumulation_plan, *plans.values())
            if candidate is not None
            for task in candidate._costed_program.selected_tasks(candidate.selections)
        ]
        streams = ResourceStreams(
            torch.device("cuda", device), (task.resource for task, _ in selected)
        )
        # Retained library allocations must precede fixed-arena admission.
        # Framework-only communication streams do not need a BLAS workspace.
        streams.initialize_blas(
            task.resource
            for task, candidate in selected
            if task.profile_id in candidate.program.executables
        )
        memory, bridge = open_path(runtime, plan, device)
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
                distributed=distributed,
            )
        )
        owner = memory.plan_handle
        if other is not None:
            memory, bridge = open_path(runtime, other, device, slab_owner=owner)
            paths.append(
                _bind(
                    runtime,
                    other,
                    memory,
                    bridge,
                    shared=paths[0].storage,
                    distributed=distributed,
                )
            )
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
                distributed=distributed,
            )
        installed = installed_runtime()
        assert installed is not None
        candidates = [path.plan for path in (*paths, *named.values())]
        layouts = []
        for candidate in candidates:
            assert candidate.memory_layout is not None
            layouts.append(candidate.memory_layout)
        seal_worker_budget(
            installed,
            selected_task_count=sum(
                len(p._costed_program.selected_tasks(p.selections)) for p in candidates
            ),
            initial_transfer_count=sum(
                r.location.value == "device"
                for p in candidates
                for r in p.schedule.initial_residency
            ),
            scheduled_transfer_count=sum(
                a.kind in {MemoryActionKind.FETCH, MemoryActionKind.EVICT}
                for p in candidates
                for a in p.schedule.actions
            ),
            fixed_lifetime_count=sum(len(layout.placements) for layout in layouts),
            dynamic_lifetime_count=sum(
                len(layout.dynamic_lifetimes) for layout in layouts
            ),
        )
        return runtime, tuple(paths), named, streams, distributed
    except BaseException:
        for path in reversed([*paths, *named.values()]):
            path.close(runtime)
        if distributed is not None:
            distributed.close()
        runtime.close()
        raise
