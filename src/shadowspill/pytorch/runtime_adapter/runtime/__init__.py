"""The process-lifetime runtime the PyTorch frontend drives, by concern.

`core` is the `Runtime` object: one per process, owning the installed allocator,
the pools and routes, the counters that say who holds it open, and the close.
`configuration` checks what a caller asks of it -- pools and routes at
construction, budgets and the device at planning. `plan` moves a plan through
its life on the runtime; `objects` names, registers and references the runtime
objects and counts persistent state; `failure` latches what a failed call left
and makes teardown safe. `calibration` measures the transfer routes and reads
the matrix the runtime publishes. `occupancy`, `retainers` and `residue` read
what a pool holds: the ranges, the framework objects over them, and what a
closing plan left behind. The submodules are the runtime's own and read its
state directly; everything else reaches the runtime through this package.
"""

from shadowspill.runtime.topology import (
    MemoryPool,
    RuntimeRoute,
    TransferCapabilities,
    TransferProfile,
)

from .configuration import RuntimeConfigurationError
from .core import PlanState, Runtime
from .failure import mark_unusable, prepare_failure_cleanup, record_failure
from .objects import (
    acquire_object_reference,
    register_object,
    release_object_generation,
    release_persistent_state,
    require_state_operation_allowed,
    reserve_persistent_object_ids,
    reserve_runtime_object_ids,
    retain_persistent_state,
)
from .occupancy import PoolAllocation, describe_live_allocations, live_allocations
from .plan import (
    PlanMemory,
    abort_plan,
    adopt_plan,
    begin_plan,
    release_plan,
    wait_plan_idle,
)
from .residue import (
    force_release_plan_scope,
    plan_scoped_residue,
    reclaim_plan_scoped_residue,
)
from .retainers import occupants, retainers

__all__ = [
    "MemoryPool",
    "PlanMemory",
    "PlanState",
    "PoolAllocation",
    "Runtime",
    "RuntimeConfigurationError",
    "RuntimeRoute",
    "TransferCapabilities",
    "TransferProfile",
    "abort_plan",
    "acquire_object_reference",
    "adopt_plan",
    "begin_plan",
    "describe_live_allocations",
    "force_release_plan_scope",
    "live_allocations",
    "mark_unusable",
    "occupants",
    "plan_scoped_residue",
    "prepare_failure_cleanup",
    "reclaim_plan_scoped_residue",
    "record_failure",
    "register_object",
    "release_object_generation",
    "release_persistent_state",
    "release_plan",
    "require_state_operation_allowed",
    "reserve_persistent_object_ids",
    "reserve_runtime_object_ids",
    "retain_persistent_state",
    "retainers",
    "wait_plan_idle",
]
