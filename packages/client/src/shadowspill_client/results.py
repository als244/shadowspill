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
    def __init__(self, status: RequestStatus) -> None:
        assert status.failure is not None
        self.status, self.failure = status, status.failure
        super().__init__(f"{self.failure.code}: {self.failure.message}")


class TrainingResult:
    """No tensor bytes are downloaded when a compute future completes."""

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
    def __init__(self, identity: RequestIdentity) -> None:
        super().__init__()
        self.identity = identity
        self.request_id = identity.request_id
        self._acceptance: CallFuture[Acceptance] = CallFuture()

    def accepted(self) -> CallFuture[Acceptance]:
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
