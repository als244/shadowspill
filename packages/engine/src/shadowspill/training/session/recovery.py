"""Classify interrupted requests without reconstructing or replaying GPU state."""

from dataclasses import dataclass
from typing import Protocol

from shadowspill_client.protocol.results import Failure, RequestStatus

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


def fail_interrupted_requests(
    records: RecoverableRequests, state: OperationState
) -> RecoverySummary:
    """On controller restart, invalidate live state and commit explicit failures.

    Dispatch must be journaled before any worker starts a command. A durable
    running record with no terminal receipt is therefore uncertain; accepted or
    incomplete uploads never started. No checkpoint or counter is inferred from
    these records, and no worker command is issued. Even an entirely terminal
    history cannot establish that its former mutable workers still exist.
    """
    state.invalidate()
    failed = uncertain = after = 0
    while page := records.pending(after=after, limit=128):
        for status in page:
            started = status.state == "running"
            records.fail(
                status.identity.request_id,
                Failure(
                    "server_restarted",
                    "Server restarted; recover live worker state from a checkpoint.",
                    "unknown" if started else "not_started",
                    state_valid=False,
                ),
            )
            failed += 1
            uncertain += started
            after = status.identity.sequence
    return RecoverySummary(failed, uncertain)
