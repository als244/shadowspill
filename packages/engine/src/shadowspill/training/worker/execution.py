"""One invocation adapter for the existing admitted and initialized Executor."""

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import BoundWorkerCall, WorkerBindings

if TYPE_CHECKING:
    from shadowspill.execution.executor import Executor


class WorkerExecution:
    """Borrow the worker's sole Executor; do not allocate or reset model state.

    The collector owns bounded host transfer/artifact staging and must release
    every borrowed execution result before it returns. Only a portable artifact
    reference may leave this adapter. Publication and controller counters happen
    after this confirmed execution/terminal-transfer boundary.
    """

    def __init__(
        self,
        executor: "Executor",
        bindings: WorkerBindings,
        budget: BufferBudget,
        collect: Callable[[Any], ArtifactRef | None],
    ) -> None:
        self._executor, self._bindings = executor, bindings
        self._budget, self._collect = budget, collect

    def __call__(self, call: BoundWorkerCall) -> ArtifactRef | None:
        if call.command.scope != self._bindings.scope:
            raise ValueError("execution call belongs to another worker")
        if call.plan is None:
            return None
        with call.input_loader(self._budget) or nullcontext(None) as loader:
            value = self._executor.run(
                plan=None if call.plan == self._bindings.primary else call.plan,
                input_loader=loader,
                hyperparams=call.hyperparams,
                accumulate_gradients=call.command.accumulate_gradients,
                runtime_trace=call.command.request.runtime_trace,
            ).result()
            try:
                result = self._collect(value)
                self._executor.synchronize()
                return result
            finally:
                del value
