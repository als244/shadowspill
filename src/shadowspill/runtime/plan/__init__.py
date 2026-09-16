"""One plan on the runtime: its objects, its admission, and what it reports.

`bridge` is the plan bound to the runtime -- the handle, the pools it selected,
and the objects its program names. `objects` registers and binds those objects
and moves their bytes; `admission` admits the program's memory actions and the
initial allocations; `report` says what the pool holds and why a call failed;
`lifecycle` begins, adopts, waits on and releases a plan.

Framework-neutral throughout: a plan is byte ranges, ids and actions. The task
boundary, where a framework's own values are rebound onto this plan's leases, is
the frontend's, in the frontend's adapter.
"""

from .bridge import (
    EncodedTask,
    PlanObjects,
    RuntimeBridge,
    RuntimeExecutionError,
    TaskMemoryEnvelope,
    TaskPublication,
    abort_task,
    actions_by_task,
    admit_caller_acquisition,
    admit_fixed_layout,
    admit_initial_actions,
    admit_task,
    begin_runtime_trace,
    clear_tasks,
    describe_object_state,
    describe_pool_occupants,
    encode_task,
    end_and_read_runtime_trace,
    input_failure_states,
    prepare_runtime_trace,
    profile_range_begin,
    profile_range_end,
    raise_if_allocator_failed,
    seal_fixed_layout,
    set_profiler_annotations,
    statistics,
)
from .lifecycle import (
    PlanMemory,
    abort_plan,
    adopt_plan,
    begin_plan,
    release_plan,
    wait_plan_idle,
)

__all__ = [
    "EncodedTask",
    "PlanMemory",
    "PlanObjects",
    "RuntimeBridge",
    "RuntimeExecutionError",
    "TaskMemoryEnvelope",
    "TaskPublication",
    "abort_plan",
    "abort_task",
    "actions_by_task",
    "admit_caller_acquisition",
    "admit_fixed_layout",
    "admit_initial_actions",
    "admit_task",
    "adopt_plan",
    "begin_plan",
    "begin_runtime_trace",
    "clear_tasks",
    "describe_object_state",
    "describe_pool_occupants",
    "encode_task",
    "end_and_read_runtime_trace",
    "input_failure_states",
    "prepare_runtime_trace",
    "profile_range_begin",
    "profile_range_end",
    "raise_if_allocator_failed",
    "release_plan",
    "seal_fixed_layout",
    "set_profiler_annotations",
    "statistics",
    "wait_plan_idle",
]
