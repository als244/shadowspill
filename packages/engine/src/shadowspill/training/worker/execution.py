"""One invocation adapter for the existing admitted and initialized Executor."""

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from shadowspill.runtime.plan.tracing import TraceCapacities
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.results import ArtifactIssue

from .bindings import BoundWorkerCall, WorkerBindings
from .checkpoint_reader import WorkerCheckpointReader
from .checkpoint_records import RankSaved
from .checkpoint_writer import WorkerCheckpointWriter
from .protocol import WorkerArtifacts, WorkerOutputError
from .trace_files import TraceFiles

if TYPE_CHECKING:
    from shadowspill.execution.executor import Executor

    from .randomness import WorkerRandomness


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
        collect: Callable[[BoundWorkerCall, Any, BufferBudget], ArtifactRef | None],
        *,
        checkpoints: WorkerCheckpointWriter | None = None,
        restore: WorkerCheckpointReader | None = None,
        randomness: "WorkerRandomness | None" = None,
        traces: TraceFiles | None = None,
        diagnostics: BufferBudget | None = None,
    ) -> None:
        self._executor, self._bindings = executor, bindings
        self._budget, self._collect = budget, collect
        self._checkpoints = checkpoints
        self._restore = restore
        self._randomness = randomness
        self._traces, self._diagnostics = traces, diagnostics

    def prepare(self, call: BoundWorkerCall) -> BoundWorkerCall:
        if call.command.request.runtime_trace:
            if self._traces is None or self._diagnostics is None:
                raise ValueError("worker has no admitted trace publication")
            bound = self._traces.budget(call)
            if bound is not None and bound.host_bytes > self._diagnostics.max_bytes:
                raise ValueError("trace exceeds the admitted worker host envelope")
        if call.command.request.operation == "restore":
            if self._restore is None:
                raise ValueError("worker has no checkpoint loader")
            return self._restore.prepare(call)
        if call.command.request.operation == "save":
            if self._checkpoints is None:
                raise ValueError("worker has no checkpoint stream")
            return self._checkpoints.prepare(call)
        return call

    def __call__(
        self, call: BoundWorkerCall
    ) -> ArtifactRef | WorkerArtifacts | RankSaved | None:
        if call.command.scope != self._bindings.scope:
            raise ValueError("execution call belongs to another worker")
        if call.command.request.operation == "restore":
            if self._restore is None:
                raise ValueError("worker has no checkpoint loader")
            self._restore(call)
            return None
        if call.command.request.operation == "save":
            if self._checkpoints is None:
                raise ValueError("worker has no checkpoint stream")
            return self._checkpoints(call)
        if call.command.request.operation == "discard":
            self._executor.discard_gradients()
            return None
        if call.command.request.operation == "close":
            self._executor.close()
            return None
        if call.plan is None:
            if call.command.request.runtime_trace:
                return WorkerArtifacts(
                    None,
                    trace_issue=ArtifactIssue(
                        "runtime_trace_skipped", "Skipped operation executed no Plan."
                    ),
                )
            return None
        evaluation = self._randomness is not None and (
            call.command.request.operation == "forward" or call.command.skipped
        )
        with (
            self._randomness.evaluation()
            if evaluation and self._randomness is not None
            else nullcontext()
        ):
            return self._run(call)

    def _run(self, call: BoundWorkerCall) -> ArtifactRef | WorkerArtifacts | None:
        bound = None if self._traces is None else self._traces.budget(call)
        if call.command.request.runtime_trace and (
            bound is None or self._diagnostics is None
        ):
            raise ValueError("worker has no prepared trace bound")
        with (
            self._diagnostics.reserve(bound.host_bytes, timeout=0)
            if bound is not None and self._diagnostics is not None
            else nullcontext()
        ):
            return self._invoke(call, None if bound is None else bound.capacities)

    def _invoke(
        self, call: BoundWorkerCall, capacity: TraceCapacities | None
    ) -> ArtifactRef | WorkerArtifacts | None:
        value = invocation = None
        trace = issue = None
        try:
            with call.input_loader(self._budget) or nullcontext(None) as loader:
                invocation = self._executor.run(
                    plan=None if call.plan == self._bindings.primary else call.plan,
                    input_loader=loader,
                    hyperparams=call.hyperparams,
                    accumulate_gradients=call.command.accumulate_gradients,
                    runtime_trace=call.command.request.runtime_trace,
                    trace_capacity=capacity,
                )
                value = invocation.result()
            # Input spill copies are synchronous. Release their staging before
            # starting serial host-output publication.
            publication_error = None
            result = None
            # Establish completed device effects before classifying a later
            # filesystem/transport error as publication failure.
            self._executor.synchronize()
            try:
                result = self._collect(call, value, self._budget)
            except OSError as error:
                publication_error = (str(error).strip() or type(error).__name__)[
                    :1024
                ].rstrip()
                # A collector traceback can retain the borrowed GPU output
                # argument. Only a bounded message may survive this scope.
                error.__traceback__ = None
                error.__cause__ = error.__context__ = None
            if call.command.request.runtime_trace:
                assert self._traces is not None
                trace, issue = self._traces.collect(call, invocation)
        finally:
            del value, invocation
        # Transfer/cleanup failure still invalidates the worker. A normal
        # publication error alone leaves its confirmed training state live.
        self._executor.synchronize()
        if publication_error is not None:
            raise WorkerOutputError(publication_error)
        if call.command.request.runtime_trace:
            return WorkerArtifacts(result, trace, issue)
        return result
