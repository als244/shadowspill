"""Worker: fail a call in a full execution pool, then close once.

Caller-owned outputs are ordinary pool allocations outside the planned slice,
so occupying every free range leaves the call's output nowhere to go: its
allocation is refused inside the task with nothing left to release.
"""

import sys
from pathlib import Path

import torch

from shadowspill import Executor, Plan
from shadowspill.status import Status

plan = Plan.load(Path(sys.argv[1]) / "plan.ssp")
executor = Executor(plan)
executor.initialize(lambda request, chunk: chunk.zero_())
held = []
while (free := executor.runtime.pool_statistics().largest_free_range_bytes) >= 1 << 20:
    held.append(torch.empty(free - (512 << 10), dtype=torch.uint8, device="cuda"))
try:
    executor.run(args=[(torch.randn(1024, 64),)]).result()
except torch.OutOfMemoryError as error:
    failure = error
else:
    raise AssertionError("a call whose output had no room in the pool succeeded")
lines = str(failure).splitlines()
names = plan.program.bindings.get("task_names", {})
labels = {names.get(task.task_id, task.task_id) for task in plan.program.tasks}
reported = [
    line.removeprefix("semantic_task: ")
    for line in lines
    if line.startswith("semantic_task: ")
]
assert reported and set(reported) <= labels, lines
assert not any(line.startswith("execution_task:") for line in lines), lines
held.clear()

# The latched runtime refuses the plan's own steps; close releases everything
# anyway and returns, since the call that failed has raised already.
executor.close()
assert executor.runtime._closed and not executor.runtime._published()
assert not executor.runtime._active_plan_handles
try:
    torch.empty((1024,), dtype=torch.uint8, device="cuda")
except RuntimeError as error:
    if f"status: {int(Status.CLOSED)} (runtime is closed)" not in str(error):
        raise
else:
    raise AssertionError("the closed allocator accepted a device allocation")
print("Closed once after an out-of-memory failure")
