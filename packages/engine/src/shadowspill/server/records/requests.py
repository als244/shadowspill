"""Durable request identities and outcomes, with a rebuildable query cache.

This stores lifecycle facts. The training coordinator still owns contiguous
acceptance, writer fencing, dependencies and dispatch through OperationState.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from shadowspill_client.protocol.codec import digest, encode
from shadowspill_client.protocol.metrics import MetricPage
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    WriterLease,
    identifier,
)
from shadowspill_client.protocol.results import (
    Acceptance,
    ExecutionReceipt,
    Failure,
    OperationResult,
    RequestStatus,
)

from .execution_receipts import validate_acceptance, validate_outcome, validate_receipt
from .journal import Journal
from .journal_format import JournalLimits, JournalRecord
from .request_index import RequestIndex


@dataclass(frozen=True)
class RunIdentity:
    owner_id: str
    session_id: str
    run_id: str

    def __post_init__(self) -> None:
        for name in ("owner_id", "session_id", "run_id"):
            identifier(getattr(self, name), name)

    def to_dict(self) -> dict[str, str]:
        return {
            name: getattr(self, name) for name in ("owner_id", "session_id", "run_id")
        }


class WriterFenced(PermissionError):
    """Authenticated run access does not authorize this retired writer lease."""


class RequestRecords:
    """Single-writer records for one owner/session/run, not live GPU-state recovery.

    The cache root is configured node-local storage. Every open builds a fresh
    disk index by streaming the authoritative journal; no shared SQLite WAL or
    unbounded in-memory request history is required. Known IDs never become new
    submissions. Accepted unfinished records stay inspectable until the request
    coordinator explicitly classifies restart failure; nothing is replayed here.
    """

    def __init__(
        self,
        directory: Path,
        *,
        identity: RunIdentity,
        cache_root: Path,
        create: bool = True,
    ):
        self.identity = identity
        self._mutex = threading.RLock()
        self._broken = self._closed = self._closing = False
        self._writer: WriterLease | None = None
        if not create and not directory.is_dir():
            raise FileNotFoundError("registered run request journal is missing")
        # A maximum-size portable request also carries its digest and journal
        # wrapper. Reserve that envelope before writing any authoritative data.
        self._journal = Journal(directory, limits=JournalLimits(record_bytes=2 << 20))
        try:
            self._index = RequestIndex(cache_root)
            rows = self._journal.records()
            first = next(rows, None)
            if first is None:
                if not create:
                    raise ValueError("request journal has no committed registration")
                self._journal.append([("run.open", identity.to_dict())])
            elif first.kind != "run.open" or dict(first.data) != identity.to_dict():
                raise ValueError("journal belongs to another owner/session/run")
            for record in rows:
                self._apply(record)
        except BaseException:
            if hasattr(self, "_index"):
                self._index.close()
            self._journal.close()
            raise

    @property
    def writer(self) -> WriterLease | None:
        with self._mutex:
            self._usable()
            return self._writer

    def claim_writer(self, writer_id: str, server_incarnation: str) -> WriterLease:
        """Transfer ownership durably; previously accepted work keeps its identity.

        Unaccepted old uploads are fenced immediately. The coordinator owns their
        terminal rejection and staging cleanup, using its actual surviving state.
        """
        with self._mutex:
            self._usable()
            previous = self._index.writer(identifier(writer_id, "writer ID"))
            if previous is not None:
                if (
                    previous == self._writer
                    and previous.server_incarnation == server_incarnation
                ):
                    return previous
                raise PermissionError(
                    "writer identity is retired; use a new explicit attachment"
                )
            lease = WriterLease(
                self.identity.run_id,
                writer_id,
                1 if self._writer is None else self._writer.writer_epoch + 1,
                server_incarnation,
                self._index.last_sequence + 1,
            )
            self._commit("writer.claim", lease.to_dict())
            return lease

    @property
    def last_sequence(self) -> int:
        with self._mutex:
            self._usable()
            return self._index.last_sequence

    def get(self, request_id: str) -> tuple[OperationRequest, RequestStatus] | None:
        with self._mutex:
            self._usable()
            row = self._index.get(identifier(request_id, "request ID"))
            return None if row is None else row[:2]

    def pending(self, *, after: int = 0, limit: int = 100) -> tuple[RequestStatus, ...]:
        if (
            type(after) is not int
            or after < 0
            or type(limit) is not int
            or not 1 <= limit <= 128
        ):
            raise ValueError(
                "pending query needs a nonnegative cursor and limit 1..128"
            )
        with self._mutex:
            self._usable()
            return tuple(
                self._require(key)[1] for key in self._index.pending(after, limit)
            )

    def at_sequence(
        self, sequence: int
    ) -> tuple[OperationRequest, RequestStatus] | None:
        if type(sequence) is not int or not 1 <= sequence < 1 << 63:
            raise ValueError("submission sequence must be a positive 63-bit integer")
        with self._mutex:
            self._usable()
            key = self._index.id_at_sequence(sequence)
            return None if key is None else self._require(key)

    def metrics(
        self, *, after: int = 0, through: int | None = None, limit: int = 100
    ) -> MetricPage:
        """Project committed outcomes; this never touches a worker or appends data."""
        from .metrics import metric_page

        with self._mutex:
            self._usable()
            return metric_page(
                self._index,
                session_id=self.identity.session_id,
                run_id=self.identity.run_id,
                position=self._journal.position,
                after=after,
                through=through,
                limit=limit,
            )

    def submit(self, request: OperationRequest) -> RequestStatus:
        """Persist metadata/ID reservation; this is explicitly not acceptance."""
        with self._mutex:
            self._usable()
            self._check_request(request)
            checksum = digest(request)
            previous = self._index.get(request.identity.request_id)
            if previous is not None:
                if checksum != previous[2]:
                    raise ValueError("request ID already names different content")
                return previous[1]
            self._check_writer(request)
            if self._index.sequence_used(request.identity.sequence):
                raise ValueError(
                    "submission sequence already belongs to another request"
                )
            self._commit(
                "request.submit", {"request": request.to_dict(), "digest": checksum}
            )
            return self._require(request.identity.request_id)[1]

    def accept(
        self, request_id: str, *, checkpoint: ArtifactRef | None = None
    ) -> RequestStatus:
        with self._mutex:
            self._usable()
            request, status = self._require(request_id)
            if status.acceptance is not None:
                return status  # Original acceptance, even after completion.
            if status.state != "uploading":
                raise ValueError("only a validated pending upload can be accepted")
            if (request.operation == "restore") != (checkpoint is not None):
                raise ValueError("restore acceptance requires its immutable checkpoint")
            self._check_writer(request)
            acceptance = Acceptance(
                status.identity, self._journal.position + 1, checkpoint
            )
            return self._status(RequestStatus(status.identity, "accepted", acceptance))

    def dispatch(self, request_id: str) -> RequestStatus:
        with self._mutex:
            self._usable()
            _, status = self._require(request_id)
            if status.state != "accepted":
                raise ValueError(
                    "dispatch requires accepted work; never replay a dispatch"
                )
            return self._status(
                RequestStatus(status.identity, "running", status.acceptance)
            )

    def complete(self, result: OperationResult) -> RequestStatus:
        with self._mutex:
            self._usable()
            request, status = self._require(result.identity.request_id)
            if result.operation != request.operation:
                raise ValueError("completed operation differs from its request")
            return self._status(
                RequestStatus(
                    result.identity, result.outcome, status.acceptance, result=result
                )
            )

    def execution(self, request_id: str) -> ExecutionReceipt | None:
        with self._mutex:
            self._usable()
            self._require(request_id)
            return self._index.execution(request_id)

    def executed(self, receipt: ExecutionReceipt) -> ExecutionReceipt:
        """Persist confirmed worker effects before publishing requested outputs."""
        with self._mutex:
            self._usable()
            request, status = self._require(receipt.identity.request_id)
            prior = self._index.execution(receipt.identity.request_id)
            if prior is not None:
                if prior != receipt:
                    raise ValueError("confirmed execution receipts are immutable")
                return prior
            validate_receipt(request, status, receipt)
            encode(receipt)
            self._commit("request.executed", receipt.to_dict())
            return receipt

    def fail(self, request_id: str, failure: Failure) -> RequestStatus:
        with self._mutex:
            self._usable()
            _, status = self._require(request_id)
            return self._status(
                RequestStatus(
                    status.identity, "failed", status.acceptance, failure=failure
                )
            )

    def _status(self, status: RequestStatus) -> RequestStatus:
        request, before = self._require(status.identity.request_id)
        if status == before:
            return before
        validate_acceptance(request, status)
        _transition(before, status)
        validate_outcome(status, self._index.execution(status.identity.request_id))
        encode(status)  # Validate the bounded portable result before durable mutation.
        self._commit("request.status", status.to_dict())
        return status

    def _commit(self, kind: str, data: dict[str, Any]) -> None:
        try:
            (record,) = self._journal.append([(kind, data)])
            self._apply(record)
        except BaseException:
            self._broken = True
            raise

    def _check_request(self, request: OperationRequest) -> None:
        if request.identity.run_id != self.identity.run_id:
            raise ValueError("request belongs to another run")
        if request.identity.sequence > (1 << 63) - 1:
            raise ValueError("submission sequence exceeds the supported 63-bit range")

    def _check_writer(self, request: OperationRequest) -> None:
        if self._writer is None or not self._writer.permits(request.identity):
            raise WriterFenced("writer lease is absent or fenced; reattach explicitly")

    def _require(self, request_id: str) -> tuple[OperationRequest, RequestStatus]:
        row = self._index.get(identifier(request_id, "request ID"))
        if row is None:
            raise KeyError(f"unknown request: {request_id}")
        return row[:2]

    def _apply(self, record: JournalRecord) -> None:
        if record.kind == "writer.claim":
            writer = WriterLease.from_dict(record.data)
            if (
                writer.run_id != self.identity.run_id
                or writer.writer_epoch
                != (1 if self._writer is None else self._writer.writer_epoch + 1)
                or writer.next_sequence != self._index.last_sequence + 1
            ):
                raise ValueError("journal writer lease epoch/run/sequence differs")
            if self._index.writer(writer.writer_id) is not None:
                raise ValueError("journal reused a retired writer identity")
            self._index.add_writer(writer)
            self._writer = writer
        elif record.kind == "request.submit":
            if set(record.data) != {"request", "digest"}:
                raise ValueError("invalid submitted request record")
            request = OperationRequest.from_dict(record.data["request"])
            self._check_request(request)
            self._check_writer(request)
            if digest(request) != record.data["digest"]:
                raise ValueError("journal request checksum differs")
            self._index.insert(
                request,
                RequestStatus(request.identity, "uploading"),
                record.data["digest"],
            )
        elif record.kind == "request.executed":
            receipt = ExecutionReceipt.from_dict(record.data)
            request, status = self._require(receipt.identity.request_id)
            validate_receipt(request, status, receipt)
            if self._index.execution(receipt.identity.request_id) is not None:
                raise ValueError("journal repeated an immutable execution receipt")
            self._index.executed(receipt)
        elif record.kind == "request.status":
            status = RequestStatus.from_dict(record.data)
            request, before = self._require(status.identity.request_id)
            validate_acceptance(request, status)
            _transition(before, status)
            validate_outcome(status, self._index.execution(status.identity.request_id))
            if (
                status.state == "accepted"
                and status.acceptance is not None
                and status.acceptance.journal_position != record.position
            ):
                raise ValueError(
                    "acceptance names a different durable journal position"
                )
            if status.state == "accepted":
                self._check_writer(request)
            if (
                status.result is not None
                and status.result.operation != request.operation
            ):
                raise ValueError("journal outcome operation differs")
            self._index.update(status, position=record.position)
        else:
            raise ValueError(f"unknown request journal event: {record.kind}")

    def _usable(self) -> None:
        if self._closed or self._closing or self._broken:
            raise RuntimeError(
                "request records closed or unavailable; reopen and reconcile"
            )

    def close(self) -> None:
        with self._mutex:
            if not self._closed:
                self._closing = True
                self._index.close()
                self._journal.close()
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _transition(before: RequestStatus, after: RequestStatus) -> None:
    allowed = {
        "uploading": {"accepted", "failed"},
        "accepted": {"running", "failed"},
        "running": {"completed", "skipped", "failed"},
    }
    if before.identity != after.identity or after.state not in allowed.get(
        before.state, set()
    ):
        raise ValueError(
            "invalid durable request transition; terminal outcomes are immutable"
        )
    if before.state != "uploading" and after.acceptance != before.acceptance:
        raise ValueError("request transition changed its original acceptance")
    if (
        before.state == "uploading"
        and after.state == "failed"
        and after.acceptance is not None
    ):
        raise ValueError("rejected uploads cannot acquire an acceptance")
    if (
        before.state != "running"
        and after.failure is not None
        and after.failure.execution_effect != "not_started"
    ):
        raise ValueError("undispatched requests cannot claim execution effects")
