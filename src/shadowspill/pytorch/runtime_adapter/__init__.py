"""PyTorch's half of the task boundary.

What is left here is the part that needs PyTorch: the operators that cross a
task boundary and rebind a plan's leases onto storages. One plan on the runtime
is `shadowspill.runtime.plan`; the runtime itself is `shadowspill.runtime`.
"""

from .boundaries import (
    acquire_for_caller,
    after_task_and_update,
    before_task_and_acquire,
    publish_initial_tensor,
    rebind,
    rebind_many,
    submit_initial_actions,
    transfer_outputs_to_caller,
    wait_task_allocations,
)

__all__ = [
    "acquire_for_caller",
    "after_task_and_update",
    "before_task_and_acquire",
    "publish_initial_tensor",
    "rebind",
    "rebind_many",
    "submit_initial_actions",
    "transfer_outputs_to_caller",
    "wait_task_allocations",
]
