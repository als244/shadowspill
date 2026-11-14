"""Commit a contiguous prefix of fully validated calls, preserving dependencies."""

from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from shadowspill_client._records import integer
from shadowspill_client.protocol.requests import ComputeRequest
from shadowspill_client.protocol.results import Failure, RequestStatus

from .recovery import RecoverableRequests
from .state import OperationState


class AcceptanceRecords(RecoverableRequests, Protocol):
    @property
    def last_sequence(self) -> int: ...

    def get(self, request_id: str) -> tuple[ComputeRequest, RequestStatus] | None: ...

    def at_sequence(
        self, sequence: int
    ) -> tuple[ComputeRequest, RequestStatus] | None: ...

    def accept(self, request_id: str) -> RequestStatus: ...


@dataclass(frozen=True)
class Rejection:
    code: str
    message: str
    dependencies: tuple[str, ...] = ()


class AcceptanceOrder:
    """CPU validation completion may be out of order; acceptance never is.

    Input ownership stays with the input queue until execution/producer cleanup.
    This component keeps only bounded ready IDs, and never copies inputs, starts
    worker work or changes optimizer/accumulation counters. The dispatcher still
    checks dependencies at execution, since an accepted dependency can later fail.
    The coordinator serializes calls with worker completion and writer takeover,
    and calls advance after each completion. Rejection waits for earlier work to
    settle so its outcome describes the actual surviving operation boundary.
    """

    def __init__(
        self,
        records: AcceptanceRecords,
        state: OperationState,
        *,
        max_pending: int,
        first_sequence: int = 1,
    ) -> None:
        self._records, self._state = records, state
        self._maximum = integer(max_pending, "pending requests", minimum=1)
        self._next = integer(first_sequence, "first sequence", minimum=1)
        self._ready: dict[str, Rejection | None] = {}
        self._mutex = RLock()

    def validated(self, request_id: str) -> RequestStatus:
        """The complete upload, controls, preprocessing and coverage have passed."""
        return self._decide(request_id, None)

    def reject(self, request_id: str, *, code: str, message: str) -> RequestStatus:
        """Queue rejection of an upload and its already queued execution suffix.

        The caller cancels producers/releases owned inputs after reading failures.
        Later explicit corrected work uses a new ID and surviving contributions;
        it must not name failed requests as dependencies.
        """
        return self._decide(request_id, Rejection(code, message))

    def _decide(self, request_id: str, rejection: Rejection | None) -> RequestStatus:
        with self._mutex:
            request, status = self._require(request_id)
            if status.state != "uploading":
                return status
            if not self._next <= request.identity.sequence < self._next + self._maximum:
                raise TimeoutError("validated request window is full")
            if self._ready.get(request_id) is None:
                self._ready[request_id] = rejection
            self.advance()
            return self._require(request_id)[1]

    def advance(self) -> None:
        """Reconsider ordered acceptance/rejection after a confirmed completion."""
        with self._mutex:
            try:
                self._advance()
            except BaseException:
                self._state.invalidate()
                raise

    def _reject_suffix(self, request_id: str, rejection: Rejection) -> bool:
        request, status = self._require(request_id)
        if status.state != "uploading":
            raise ValueError("validation rejection cannot undo dispatched work")
        first = self._records.pending(limit=1)
        if first and first[0].identity.sequence < request.identity.sequence:
            return False
        # New corrected work may be submitted as soon as the first failure is
        # visible. Only the already queued suffix belongs to this rejection.
        cutoff = self._records.last_sequence
        self._fail(request_id, rejection)
        after = request.identity.sequence
        while page := self._records.pending(after=after, limit=128):
            for pending in page:
                if pending.identity.sequence > cutoff:
                    return True
                if pending.state == "running":
                    raise RuntimeError("dependent suffix was dispatched out of order")
                self._fail(
                    pending.identity.request_id,
                    Rejection(
                        "dependency_failed",
                        "An earlier queued request failed before execution.",
                        (request_id,),
                    ),
                )
                after = pending.identity.sequence
        return True

    def _advance(self) -> None:
        while (row := self._records.at_sequence(self._next)) is not None:
            request, status = row
            request_id = request.identity.request_id
            if status.state != "uploading":
                self._ready.pop(request_id, None)
                self._next += 1
                continue
            if request_id not in self._ready:
                break
            rejection = self._ready[request_id] or self._invalid(request)
            if rejection is not None:
                if not self._reject_suffix(request_id, rejection):
                    break
                continue
            try:
                self._records.accept(request_id)
            except PermissionError:
                self._ready[request_id] = Rejection(
                    "writer_fenced",
                    "The upload belongs to a fenced writer.",
                )
                continue
            self._ready.pop(request_id)
            self._next += 1

    def _invalid(self, request: ComputeRequest) -> Rejection | None:
        if not self._state.boundary.valid or self._state.boundary.closed:
            return Rejection(
                "state_unavailable", "Live training state is invalid or closed."
            )
        failed, invalid = [], []
        for dependency in request.dependencies:
            prior = self._records.get(dependency)
            if prior is None or prior[0].identity.sequence >= request.identity.sequence:
                invalid.append(dependency)
            elif prior[1].state == "failed":
                failed.append(dependency)
        if failed or invalid:
            return Rejection(
                "dependency_failed" if failed else "invalid_dependency",
                "Dependencies must name earlier requests that have not failed.",
                tuple(failed or invalid),
            )
        return None

    def _fail(self, request_id: str, rejection: Rejection) -> None:
        boundary = self._state.boundary
        failure = Failure(
            rejection.code,
            rejection.message,
            "not_started",
            boundary.valid,
            completed_updates=boundary.completed_updates,
            accumulation=boundary.accumulation,
            failed_dependencies=rejection.dependencies,
        )
        self._records.fail(request_id, failure)
        self._ready.pop(request_id, None)

    def _require(self, request_id: str) -> tuple[ComputeRequest, RequestStatus]:
        result = self._records.get(request_id)
        if result is None:
            raise KeyError(f"unknown request: {request_id}")
        return result
