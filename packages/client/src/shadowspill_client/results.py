"""Training outcomes retain their durable record and explicit output handles."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from types import MappingProxyType
from typing import TYPE_CHECKING

from ._async import CallFuture
from .checkpoints import CheckpointHandle
from .protocol.requests import RequestIdentity
from .protocol.results import Acceptance, OperationResult, RequestStatus
from .traces import TraceHandle
from .weights import WeightsHandle

if TYPE_CHECKING:
    from .service import ServiceClient


class TrainingRequestError(RuntimeError):
    """A training request the service reports as failed.

    The message is the failure's code and message.

    Parameters
    ----------
    status : RequestStatus
        The request's final status; its ``failure`` is set.

    Attributes
    ----------
    failure : Failure
        The service's failure record.
    """

    def __init__(self, status: RequestStatus) -> None:
        assert status.failure is not None
        self.status, self.failure = status, status.failure
        super().__init__(f"{self.failure.code}: {self.failure.message}")


class TrainingResult:
    """The outcome of a finished training request.

    The futures of `TrainingClient.step`, `TrainingClient.forward_backward`,
    `TrainingClient.optim_step`, `TrainingClient.forward`,
    `TrainingClient.discard_gradients`, `TrainingClient.load_checkpoint` and
    `TrainingClient.close` resolve to one, as does `RequestQueries.get` for
    any request of the run, including a checkpoint save or weights export.
    It downloads nothing: tensor outputs, traces, checkpoints and exported
    weights are handles, each downloaded explicitly.

    Attributes
    ----------
    identity : RequestIdentity
        The request's identity.
    request_id : str
        The request's ID.
    operation : str
        ``"step"``, ``"forward_backward"``, ``"optim_step"``, ``"forward"``,
        ``"discard"``, ``"save"``, ``"restore"`` or ``"close"``.
    outcome : str
        Always ``"completed"``.
    history_id : str
        The training history the request belongs to. Loading a checkpoint
        starts a new one.
    completed_updates : int
        Optimizer updates the run has completed, this request included; after
        a checkpoint load, the saved count.
    accumulation : Accumulation
        The gradient accumulation after this request: ``open`` is True while
        gradients wait for `TrainingClient.optim_step`, ``contributions``
        lists the IDs of the requests that added to it, and ``generation``
        counts accumulations.
    state_valid : bool
        Whether the run's training state is usable after this request. Only
        ``close`` can finish with False.
    loss : float or None
        The objective's loss, summed over the batch with no mean taken; None
        when the operation computes none.
    metrics : mapping of str to float
        The objective's named metrics, each combined over the batch as the
        objective declares.
    execution_seconds : float or None
        Execution time of the slowest worker, in seconds, from input staging
        to the final synchronization. It excludes upload and queueing. None
        for state operations, or when a worker reported no time.
    optim : mapping of str to scalar
        Every optimizer control the update used, defaults included; empty
        unless the operation is ``"step"`` or ``"optim_step"``.
    loss_params : mapping of str to scalar
        Every objective control the operation used, defaults included; empty
        for ``"optim_step"`` and state operations.
    grad_scale : float or None
        The gradient scale the update applied; None unless the operation is
        ``"step"`` or ``"optim_step"``.
    metadata : mapping of str to object
        For ``"restore"``, the caller metadata saved with the checkpoint;
        empty otherwise.
    outputs : mapping of str to ArtifactHandle
        The output set's tensor outputs, by name.
    traces : tuple of TraceHandle
        Runtime traces, one per worker, when the call set ``runtime_trace``.
    artifact_issues : tuple of ArtifactIssue
        Optional artifacts, such as traces, that could not be produced. The
        operation itself succeeded and must not be repeated.
    checkpoint : CheckpointHandle or None
        The checkpoint saved by `TrainingClient.save_checkpoint` or loaded by
        `TrainingClient.load_checkpoint`; None otherwise.
    weights : WeightsHandle or None
        The weights exported by `TrainingClient.export_weights`; None
        otherwise.
    record : OperationResult
        The service's durable record of the outcome, which the other
        attributes present; it holds artifact references in place of handles.
    """

    def __init__(self, record: OperationResult, service: ServiceClient) -> None:
        self.record = record
        self.identity, self.operation = record.identity, record.operation
        self.request_id = record.identity.request_id
        self.history_id, self.outcome = record.history_id, record.outcome
        self.completed_updates = record.completed_updates
        self.accumulation, self.state_valid = record.accumulation, record.state_valid
        self.loss, self.metrics = record.loss, record.metrics
        self.execution_seconds = record.execution_seconds
        self.metadata = record.metadata
        self.optim, self.loss_params, self.grad_scale = (
            record.optim,
            record.loss_params,
            record.grad_scale,
        )
        self.artifact_issues = record.artifact_issues
        self.outputs = MappingProxyType(
            {
                name: service.artifact(reference)
                for name, reference in record.outputs.items()
            }
        )
        self.traces = tuple(
            TraceHandle(reference, service) for reference in record.traces
        )
        self.checkpoint = (
            None
            if record.checkpoint is None
            else CheckpointHandle(record.checkpoint, service)
        )
        self.weights = (
            None if record.weights is None else WeightsHandle(record.weights, service)
        )


class OperationFuture[T](CallFuture[T]):
    """The pending result of one submitted training request.

    Every `TrainingClient` operation, such as `TrainingClient.step`, returns
    one. It fails with `TrainingRequestError` when the service reports the
    request failed. It fails with the exchange's own error, such as
    `TransportError`, when an exchange with the service fails; the request may
    then still run, so look it up by ``request_id`` instead of submitting it
    again.

    Attributes
    ----------
    identity : RequestIdentity
        The request's identity, fixed at submission.
    request_id : str
        The request's ID, for `RequestQueries` lookups after a reconnect.
    """

    def __init__(self, identity: RequestIdentity) -> None:
        super().__init__()
        self.identity = identity
        self.request_id = identity.request_id
        self._acceptance: CallFuture[Acceptance] = CallFuture()

    def accepted(self) -> CallFuture[Acceptance]:
        """Return a future for the service's acceptance of this request.

        The service accepts requests in submission order, once a request's
        input has arrived and passed the service's checks. An accepted
        request goes on even if this client disconnects.

        Returns
        -------
        CallFuture of Acceptance
            Resolves to the acceptance. It fails with the request's own error
            if the request fails before it is accepted.
        """
        return self._acceptance

    def _mapped[U](self, convert: Callable[[T], U]) -> OperationFuture[U]:
        """Adapt a completed result without another submission or queued task."""
        result = OperationFuture[U](self.identity)
        result._acceptance = self._acceptance

        def complete(source: Future[T]) -> None:
            try:
                value = convert(source.result())
            except BaseException as error:
                error.__traceback__ = None
                error.__cause__ = error.__context__ = None
                result._set_exception(error)
            else:
                result._set_result(value)

        self._future.add_done_callback(complete)
        return result

    def _observe(self, status: RequestStatus) -> None:
        if status.identity != self.identity:
            raise ValueError("request reply belongs to a different submission")
        if status.acceptance is not None and not self._acceptance.done():
            self._acceptance._set_result(status.acceptance)

    def _set_exception(self, error: BaseException) -> None:
        if not self._acceptance.done():
            self._acceptance._set_exception(error)
        super()._set_exception(error)
