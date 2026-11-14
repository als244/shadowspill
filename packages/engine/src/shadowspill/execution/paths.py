"""Bind and release one admitted invocation path in a worker."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.plan import Plan
from shadowspill.preparation.lifecycle import LoadedPreparations
from shadowspill.pytorch.numerics import (
    program_numerical_contracts,
    validate_numerical_settings,
)
from shadowspill.runtime import Runtime
from shadowspill.runtime.plan.admission import clear_tasks
from shadowspill.runtime.plan.bridge import RuntimeBridge
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
    numerical_contracts: tuple[str, ...] = ()
    accumulation: ExecutionPath | None = None

    def close(self, runtime: Runtime) -> None:
        # Each step runs whatever the ones before it raised, last registered
        # first: a runtime that latched a failure refuses the task and object
        # steps, and the plan handle is still released.
        objects = self.bridge.objects
        with ExitStack() as cleanup:
            cleanup.callback(release_plan, runtime, self.memory.plan_handle)
            cleanup.callback(objects.unregister, objects.registered_aliases())
            cleanup.callback(clear_tasks, self.bridge)
            cleanup.callback(self.storage.clear_views)


def plan_numerical_contracts(plan: Plan) -> tuple[str, ...]:
    selected = {
        task.task_id for task in plan._costed_program.selected_tasks(plan.selections)
    }
    return program_numerical_contracts(
        plan.program,
        {task.profile_id for task in plan.program.tasks if task.task_id in selected},
    )


def bind_path(
    runtime: Runtime,
    plan: Plan,
    memory: PlanMemory,
    bridge: RuntimeBridge,
    *,
    shared_sources: Mapping[str, tuple[Storage, str]] | None = None,
    shared_host: Mapping[str, torch.Tensor] | None = None,
    extra_initial: tuple[str, ...] = (),
    shareable_aliases: tuple[str, ...] = (),
    shared_slice_bytes: int | None = None,
    distributed: DistributedWorker | None = None,
    preparations: LoadedPreparations | None = None,
) -> ExecutionPath:
    try:
        contracts = plan_numerical_contracts(plan)
        validate_numerical_settings(contracts)
        require_compiled_device(plan.program, memory.execution_device)
        storage = Storage(
            plan,
            bridge,
            torch.device("cuda", memory.execution_device),
            shared_sources=shared_sources,
            extra_initial=extra_initial,
            shareable_aliases=shareable_aliases,
            logical_rank=None
            if distributed is None
            else distributed.groups.logical_rank,
            transfer_descriptors=None
            if distributed is None
            else distributed.descriptors(plan),
        )
        verify_initialization(storage)
        controls = Controls(storage, shared_host=shared_host)
        tasks, acquisition = prepare(
            plan,
            bridge,
            shared_slice_bytes=shared_slice_bytes,
            groups=None if distributed is None else distributed.groups,
            preparations=preparations,
        )
        adopt_plan(runtime, memory.plan_handle)
        return ExecutionPath(
            plan, memory, bridge, storage, controls, tasks, acquisition, contracts
        )
    except BaseException:
        bridge.objects.unregister(bridge.objects.registered_aliases())
        abort_plan(runtime, memory.plan_handle)
        raise
