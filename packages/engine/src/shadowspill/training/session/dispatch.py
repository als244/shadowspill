"""One ordered dispatch boundary around the existing operation state machine."""

from collections.abc import Callable
from threading import Lock
from typing import Protocol

from shadowspill_client.protocol.checkpoints import CHECKPOINT_MEDIA_TYPE
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, ProtocolError, encode
from shadowspill_client.protocol.requests import ArtifactRef, OperationRequest
from shadowspill_client.protocol.results import (
    Acceptance,
    ExecutionReceipt,
    Failure,
    OperationResult,
    RequestStatus,
)

from .acceptance import AcceptanceOrder, AcceptanceRecords
from .checkpoint import ResolvedCheckpoint
from .state import OperationState, Transition


class DispatchRecords(AcceptanceRecords, Protocol):
    def dispatch(self, request_id: str) -> RequestStatus: ...

    def executed(self, receipt: ExecutionReceipt) -> ExecutionReceipt: ...

    def complete(self, result: OperationResult) -> RequestStatus: ...


class CompletedWork(Protocol):
    """Owned host outputs after every worker and required transfer has completed.

    A worker adapter returns this only after its completion checks. It must not
    retain execution-buffer views behind publication handles. The owner is closed
    even when authoritative storage or required publication fails.
    """

    def close(self) -> None: ...


class ExecutionNotStarted(RuntimeError):
    """All workers confirmed unstarted work and still own valid mutable state.

    A worker adapter may raise this only after every staging rejection/abort is
    acknowledged and no execute was sent. Missing replies, worker loss and any
    error after release must use the ordinary unknown-effect failure path.
    """


class RequestDispatcher:
    """Dispatch only the accepted queue head; never retry uncertain worker work.

    The caller provides the prepared worker invocation and bounded publisher.
    Neither input validation, model execution nor tensor serialization lives here.
    Validation can overlap worker execution through ``order``; brief record/state
    changes share its lock. One dispatch includes receipt and publication commit.
    """

    def __init__(
        self,
        records: DispatchRecords,
        state: OperationState,
        *,
        max_pending: int,
        first_sequence: int = 1,
    ) -> None:
        self._records, self._state = records, state
        self.order = AcceptanceOrder(
            records, state, max_pending=max_pending, first_sequence=first_sequence
        )
        self._dispatching = Lock()

    def run[Work: CompletedWork](
        self,
        request_id: str,
        execute: Callable[[Transition], Work],
        publish: Callable[[ExecutionReceipt, Work], OperationResult],
        *,
        participates: bool | None = None,
        restore: ResolvedCheckpoint | None = None,
    ) -> RequestStatus:
        """Synchronous worker boundary; the service runs it outside HTTP handlers.

        ``execute`` must confirm every participating worker and terminal transfer
        before returning. Only ``ExecutionNotStarted`` certifies intact workers
        before release; other exceptions have unknown effects. ``publish``
        performs no model work; failure cannot roll back or rerun execution.
        """
        if not self._dispatching.acquire(blocking=False):
            raise RuntimeError("another request is being dispatched")
        try:
            with self.order.synchronized():
                started = self._start(request_id, participates, restore)
            if isinstance(started, RequestStatus):
                return started
            request, transition = started
            try:
                work = execute(transition)
            except ExecutionNotStarted as error:
                with self.order.synchronized():
                    boundary = self._state.fail(transition, execution_started=False)
                    return self._failed(
                        request,
                        Failure(
                            "worker_preflight_failed",
                            str(error) or type(error).__name__,
                            "not_started",
                            boundary.valid,
                            boundary.completed_updates,
                            boundary.accumulation,
                            history_id=boundary.history_id,
                        ),
                    )
            except BaseException as error:
                with self.order.synchronized():
                    self._state.fail(transition, execution_started=True)
                    status = self._failed(
                        request,
                        Failure(
                            "execution_failed",
                            str(error) or type(error).__name__,
                            "unknown",
                            False,
                        ),
                    )
                if not isinstance(error, Exception):
                    raise
                return status
            try:
                with self.order.synchronized():
                    receipt = self._confirm(request, transition, restore)
                return self._publish(request, receipt, work, publish)
            finally:
                try:
                    work.close()
                except BaseException:
                    with self.order.synchronized():
                        self._state.invalidate()
                    raise
        finally:
            self._dispatching.release()

    def _start(
        self,
        request_id: str,
        participates: bool | None,
        restore: ResolvedCheckpoint | None,
    ) -> tuple[OperationRequest, Transition] | RequestStatus:
        row = self._records.get(request_id)
        if row is None:
            raise KeyError(f"unknown request: {request_id}")
        request, status = row
        if status.state in ("completed", "skipped", "failed"):
            return status
        if (request.operation == "restore") != (restore is not None) or (
            restore is not None
            and (
                status.acceptance is None
                or status.acceptance.checkpoint != restore.reference
            )
        ):
            raise ValueError("dispatch checkpoint differs from its accepted input")
        first = self._records.pending(limit=1)
        if (
            status.state != "accepted"
            or not first
            or first[0].identity != request.identity
        ):
            raise RuntimeError("dispatch requires the accepted execution queue head")
        # An accepted predecessor can fail after this request was accepted.
        for dependency in request.dependencies:
            prior = self._records.get(dependency)
            if prior is None or prior[1].state not in ("completed", "skipped"):
                return self.order.reject(
                    request_id,
                    code="dependency_failed",
                    message="An earlier dependency did not complete successfully.",
                    dependencies=(dependency,),
                )
        try:
            transition = self._state.begin(
                request.operation, request_id, participates=participates
            )
        except (TypeError, ValueError, RuntimeError) as error:
            return self.order.reject(
                request_id, code="operation_conflict", message=str(error)
            )
        try:
            assert status.acceptance is not None
            self._preflight(request, transition, status.acceptance, restore)
        except ProtocolError:
            self._state.fail(transition, execution_started=False)
            return self.order.reject(
                request_id,
                code="metadata_limit",
                message="Completion metadata exceeds the portable record bound; "
                "reduce metadata or finish the current accumulation.",
            )
        try:
            self._records.dispatch(request_id)
        except BaseException:
            self._state.invalidate()
            raise
        return request, transition

    def _preflight(
        self,
        request: OperationRequest,
        transition: Transition,
        acceptance: Acceptance,
        restore: ResolvedCheckpoint | None,
    ) -> None:
        """Reserve the contribution-history envelope before a mutating invocation.

        The prepared publisher separately admits its metric/artifact descriptors.
        This check covers the known counters, request controls and worst-length
        history ID, so growing accumulation cannot overflow the execution receipt.
        """
        after = self._state.completion(
            transition,
            restored_updates=None
            if restore is None
            else restore.manifest.info.completed_updates,
        )
        result = OperationResult(
            request.identity,
            request.operation,
            "h" * 128,
            "skipped" if transition.skipped else "completed",
            after.completed_updates,
            after.accumulation,
            optim=request.optim,
            loss_params=request.loss_params,
            grad_scale=request.grad_scale,
            state_valid=after.valid,
            checkpoint=ArtifactRef(
                "c" * 128, "0" * 64, MAX_METADATA_BYTES, CHECKPOINT_MEDIA_TYPE
            )
            if request.operation in ("save", "restore")
            else None,
            metadata={} if restore is None else restore.manifest.info.metadata,
        )
        encode(ExecutionReceipt.from_result(result))
        encode(
            RequestStatus(request.identity, result.outcome, acceptance, result=result)
        )

    def _confirm(
        self,
        request: OperationRequest,
        transition: Transition,
        restore: ResolvedCheckpoint | None,
    ) -> ExecutionReceipt:
        try:
            boundary = self._state.complete(
                transition,
                restored_updates=None
                if restore is None
                else restore.manifest.info.completed_updates,
            )
            receipt = ExecutionReceipt(
                request.identity,
                request.operation,
                boundary.history_id,
                "skipped" if transition.skipped else "completed",
                boundary.completed_updates,
                boundary.accumulation,
                state_valid=boundary.valid,
            )
            self._records.executed(receipt)
            return receipt
        except BaseException:
            self._state.invalidate()
            raise

    def _publish[Work: CompletedWork](
        self,
        request: OperationRequest,
        receipt: ExecutionReceipt,
        work: Work,
        publish: Callable[[ExecutionReceipt, Work], OperationResult],
    ) -> RequestStatus:
        try:
            result = publish(receipt, work)
            if ExecutionReceipt.from_result(result) != receipt:
                raise ValueError("published result differs from confirmed execution")
        except BaseException as error:
            with self.order.synchronized():
                status = self._failed(
                    request,
                    Failure(
                        "publication_failed",
                        str(error) or type(error).__name__,
                        "completed",
                        self._state.boundary.valid,
                        receipt.completed_updates,
                        receipt.accumulation,
                        history_id=receipt.history_id,
                    ),
                )
            if not isinstance(error, Exception):
                raise
            return status
        with self.order.synchronized():
            try:
                status = self._records.complete(result)
                self.order.advance()
                return status
            except BaseException:
                self._state.invalidate()
                raise

    def _failed(self, request: OperationRequest, failure: Failure) -> RequestStatus:
        try:
            status = self._records.fail(request.identity.request_id, failure)
            self.order.fail_dependents(request.identity.request_id)
            return status
        except BaseException:
            self._state.invalidate()
            raise
