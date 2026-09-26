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
from shadowspill.runtime import Runtime
from shadowspill.runtime.abi import runtime_library
from shadowspill.runtime.failures import RuntimeExecutionError

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
from .common import (
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
    describe_refused_action,
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

    def wait_until_idle(self) -> None:
        """Wait for the work this plan owns, and no one else's."""

        self.require(
            self.runtime_library.shadowspill_plan_wait_idle(self.plan_handle),
            "wait for plan idle",
        )

    def wait_runtime_idle(self) -> None:
        """Wait for runtime-global quiescence, at a lifecycle boundary."""

        self.require(
            self.runtime_library.shadowspill_runtime_wait_idle(
                self.runtime._runtime_handle
            ),
            "wait idle",
        )

    def require_empty_layout(self) -> None:
        """Refuse unless nothing is live in this plan's layout.

        Asked at the start of a call, once every plan placing into the layout
        has drained: whatever an earlier call placed there is gone by then, so
        a survivor is something this call would overwrite.
        """

        self.require(
            self.runtime_library.shadowspill_plan_require_empty_layout(
                self.plan_handle
            ),
            "check that the plan's layout is empty between calls",
        )

    def require(self, raw_status: Any, operation: str) -> None:
        """Raise for a nonzero adapter status, naming the operation."""

        require_status(self.library, raw_status, operation)




def abort_task(bridge: RuntimeBridge, task_handle: int) -> None:
    """Close the matching admitted task scope after frontend failure."""

    bridge.require(
        bridge.library.shadowspill_pytorch_abort_task_handle(task_handle),
        "abort admitted task",
    )


__all__ = [
    "EncodedTask",
    "PlanObjects",
    "RuntimeBridge",
    "RuntimeExecutionError",
    "TaskMemoryEnvelope",
    "TaskPublication",
    "abort_task",
    "actions_by_task",
    "admit_caller_acquisition",
    "admit_fixed_layout",
    "admit_initial_actions",
    "admit_task",
    "begin_runtime_trace",
    "clear_tasks",
    "describe_object_state",
    "describe_pool_occupants",
    "describe_refused_action",
    "encode_task",
    "end_and_read_runtime_trace",
    "input_failure_states",
    "prepare_runtime_trace",
    "profile_range_begin",
    "profile_range_end",
    "raise_if_allocator_failed",
    "seal_fixed_layout",
    "set_profiler_annotations",
    "statistics",
]
