"""Training outcomes retain their durable record and explicit output handles."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from ._async import CallFuture
from .protocol.requests import RequestIdentity
from .protocol.results import Acceptance, OperationResult, RequestStatus

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
        self.history_id, self.outcome = record.history_id, record.outcome
        self.completed_updates = record.completed_updates
        self.accumulation, self.state_valid = record.accumulation, record.state_valid
        self.loss, self.metrics = record.loss, record.metrics
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
        self.traces = tuple(service.artifact(reference) for reference in record.traces)


class OperationFuture(CallFuture[TrainingResult]):
    def __init__(self, identity: RequestIdentity) -> None:
        super().__init__()
        self.identity = identity
        self.request_id = identity.request_id
        self._acceptance: CallFuture[Acceptance] = CallFuture()

    def accepted(self) -> CallFuture[Acceptance]:
        return self._acceptance

    def _observe(self, status: RequestStatus) -> None:
        if status.identity != self.identity:
            raise ValueError("request reply belongs to a different submission")
        if status.acceptance is not None and not self._acceptance.done():
            self._acceptance._set_result(status.acceptance)

    def _set_exception(self, error: BaseException) -> None:
        if not self._acceptance.done():
            self._acceptance._set_exception(error)
        super()._set_exception(error)
