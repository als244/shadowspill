"""The bridge: one plan's program on the adapter, from admission to close.

`RuntimeBridge` holds what every part needs -- the runtime and its two
libraries, the plan handle, the pools the plan chose, the plan's object
registry (`objects`) and the records of what was admitted. The work is in the
modules beside it, as functions over the bridge: `admission` describes tasks,
the fixed layout, action batches and acquisitions to the runtime before the
plan runs; `boundaries` is what happens at each boundary while it runs;
`report` is what the bridge can say about the plan without changing it;
`objects` is the registry of plan-local aliases bound to runtime objects; and
`common` the values and helpers they share.
"""

from __future__ import annotations

from typing import Any

from shadowspill.ir import ShadowSpillProgram
from shadowspill.pytorch.runtime_adapter.abi import runtime_library
from shadowspill.pytorch.runtime_adapter.failures import RuntimeExecutionError
from shadowspill.pytorch.runtime_adapter.runtime import Runtime

from .admission import (
    EncodedTask,
    admit_caller_acquisition,
    admit_fixed_layout,
    admit_initial_actions,
    admit_task,
    clear_tasks,
    encode_task,
    seal_fixed_layout,
)
from .boundaries import (
    abort_task,
    acquire_for_caller,
    after_task_and_update,
    before_task_and_acquire,
    dematerialize,
    publish_initial_tensor,
    rebind,
    rebind_many,
    submit_initial_actions,
    transfer_outputs_to_caller,
    wait_idle,
    wait_plan_idle,
    wait_task_allocations,
)
from .common import (
    PublishedStorage,
    TaskMemoryEnvelope,
    TaskPublication,
    actions_by_task,
    require_status,
)
from .objects import PlanObjects
from .report import (
    begin_runtime_trace,
    describe_object_state,
    describe_pool_occupants,
    end_and_read_runtime_trace,
    input_failure_states,
    prepare_runtime_trace,
    profile_range_begin,
    profile_range_end,
    raise_if_allocator_failed,
    set_profiler_annotations,
    statistics,
)


class RuntimeBridge:
    """One plan's program bound to the runtime, and the records of its admission."""

    def __init__(
        self,
        runtime: Runtime,
        program: ShadowSpillProgram,
        plan_handle: int,
        *,
        execution_pool_id: int,
        spill_pool_id: int,
    ) -> None:
        if execution_pool_id < 0 or spill_pool_id < 0:
            raise ValueError("plan pool IDs must be non-negative")
        if execution_pool_id == spill_pool_id:
            raise ValueError("execution and spill pools must be distinct")
        self.runtime = runtime
        self.library = runtime._installed.library
        # Plan admission is the neutral runtime's own API, and the bridge
        # calls it directly.
        self.runtime_library = runtime_library()
        self.plan_handle = plan_handle
        self.execution_pool_id = execution_pool_id
        self.spill_pool_id = spill_pool_id
        self.objects = PlanObjects(
            runtime,
            self.runtime_library,
            program,
            plan_handle,
            spill_pool_id=spill_pool_id,
        )
        self._admitted_task_handles: set[int] = set()
        self._admitted_action_batches: dict[
            int, tuple[int, tuple[tuple[str, Any], ...]]
        ] = {}
        self._admitted_acquisitions: dict[tuple[str, ...], int] = {}
        self._fixed_layout_installed = False

    def require(self, raw_status: Any, operation: str) -> None:
        """Raise for a nonzero adapter status, naming the operation."""

        require_status(self.library, raw_status, operation)


__all__ = [
    "EncodedTask",
    "PlanObjects",
    "PublishedStorage",
    "RuntimeBridge",
    "RuntimeExecutionError",
    "TaskMemoryEnvelope",
    "TaskPublication",
    "abort_task",
    "acquire_for_caller",
    "actions_by_task",
    "admit_caller_acquisition",
    "admit_fixed_layout",
    "admit_initial_actions",
    "admit_task",
    "after_task_and_update",
    "before_task_and_acquire",
    "begin_runtime_trace",
    "clear_tasks",
    "dematerialize",
    "describe_object_state",
    "describe_pool_occupants",
    "encode_task",
    "end_and_read_runtime_trace",
    "input_failure_states",
    "prepare_runtime_trace",
    "profile_range_begin",
    "profile_range_end",
    "publish_initial_tensor",
    "raise_if_allocator_failed",
    "rebind",
    "rebind_many",
    "seal_fixed_layout",
    "set_profiler_annotations",
    "statistics",
    "submit_initial_actions",
    "transfer_outputs_to_caller",
    "wait_idle",
    "wait_plan_idle",
    "wait_task_allocations",
]
