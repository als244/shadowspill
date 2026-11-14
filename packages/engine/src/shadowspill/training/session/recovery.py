"""Classify interrupted requests without reconstructing or replaying GPU state."""

from dataclasses import dataclass
from typing import Protocol

from shadowspill_client._records import text
from shadowspill_client.protocol.results import ExecutionReceipt, Failure, RequestStatus

from .state import OperationState


class RecoverableRequests(Protocol):
    """Durable outcome interface; the training engine does not import server I/O."""

    def pending(
        self, *, after: int = 0, limit: int = 100
    ) -> tuple[RequestStatus, ...]: ...

    def fail(self, request_id: str, failure: Failure) -> RequestStatus: ...


@dataclass(frozen=True)
class RecoverySummary:
    failed_requests: int
    uncertain_executions: int
    confirmed_executions: int = 0


class InterruptedRequests(RecoverableRequests, Protocol):
    def execution(self, request_id: str) -> ExecutionReceipt | None: ...


def fail_interrupted_requests(
    records: InterruptedRequests,
    state: OperationState,
    *,
    code: str = "server_restarted",
    message: str = "Server restarted; recover live worker state from a checkpoint.",
) -> RecoverySummary:
    """After controller loss, invalidate live state and commit explicit failures.

    Dispatch must be journaled before any worker starts a command. A durable
    running record with no execution receipt is therefore uncertain; accepted or
    incomplete uploads never started. Confirmed execution retains its known
    effect despite unfinished publication. No checkpoint or counter is inferred from
    these records, and no worker command is issued. Even an entirely terminal
    history cannot establish that its former mutable workers still exist.
    """
    text(code, "recovery error code")
    text(message, "recovery error message")
    state.invalidate()
    failed = uncertain = confirmed = after = 0
    while page := records.pending(after=after, limit=128):
        for status in page:
            started = status.state == "running"
            receipt = records.execution(status.identity.request_id)
            records.fail(
                status.identity.request_id,
                Failure(
                    code,
                    message,
                    "completed" if receipt else "unknown" if started else "not_started",
                    state_valid=False,
                    completed_updates=None
                    if receipt is None
                    else receipt.completed_updates,
                    accumulation=None if receipt is None else receipt.accumulation,
                ),
            )
            failed += 1
            uncertain += started and receipt is None
            confirmed += receipt is not None
            after = status.identity.sequence
    return RecoverySummary(failed, uncertain, confirmed)
