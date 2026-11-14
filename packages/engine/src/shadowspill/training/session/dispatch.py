"""One ordered dispatch boundary around the existing operation state machine."""

from collections.abc import Callable
from threading import Lock
from typing import Protocol

from shadowspill_client.protocol.codec import ProtocolError, encode
from shadowspill_client.protocol.requests import ComputeRequest
from shadowspill_client.protocol.results import (
    Acceptance,
    ComputeResult,
    ExecutionReceipt,
    Failure,
    RequestStatus,
)

from .acceptance import AcceptanceOrder, AcceptanceRecords
from .state import OperationState, Transition


class DispatchRecords(AcceptanceRecords, Protocol):
    def dispatch(self, request_id: str) -> RequestStatus: ...

    def executed(self, receipt: ExecutionReceipt) -> ExecutionReceipt: ...

    def complete(self, result: ComputeResult) -> RequestStatus: ...


class CompletedWork(Protocol):
    """Owned host outputs after every worker and required transfer has completed.

    A worker adapter returns this only after its completion checks. It must not
    retain execution-buffer views behind publication handles. The owner is closed
    even when authoritative storage or required publication fails.
    """

    def close(self) -> None: ...


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
        publish: Callable[[ExecutionReceipt, Work], ComputeResult],
        *,
        participates: bool | None = None,
    ) -> RequestStatus:
        """Synchronous worker boundary; the service runs it outside HTTP handlers.

        ``execute`` must confirm every participating worker and terminal transfer
        before returning. Any exception there has unknown effects. ``publish``
        performs no model work; failure cannot roll back or rerun execution.
        """
        if not self._dispatching.acquire(blocking=False):
            raise RuntimeError("another request is being dispatched")
        try:
            with self.order.synchronized():
                started = self._start(request_id, participates)
            if isinstance(started, RequestStatus):
                return started
            request, transition = started
            try:
                work = execute(transition)
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
                    receipt = self._confirm(request, transition)
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
        self, request_id: str, participates: bool | None
    ) -> tuple[ComputeRequest, Transition] | RequestStatus:
        row = self._records.get(request_id)
        if row is None:
            raise KeyError(f"unknown request: {request_id}")
        request, status = row
        if status.state in ("completed", "skipped", "failed"):
            return status
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
            self._preflight(request, transition, status.acceptance)
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
        request: ComputeRequest,
        transition: Transition,
        acceptance: Acceptance,
    ) -> None:
        """Reserve the contribution-history envelope before a mutating invocation.

        The prepared publisher separately admits its metric/artifact descriptors.
        This check covers the known counters, request controls and worst-length
        history ID, so growing accumulation cannot overflow the execution receipt.
        """
        after = self._state.completion(transition)
        result = ComputeResult(
            request.identity,
            request.operation,
            "h" * 128,
            "skipped" if transition.skipped else "completed",
            after.completed_updates,
            after.accumulation,
            optim=request.optim,
            loss_params=request.loss_params,
            grad_scale=request.grad_scale,
        )
        encode(ExecutionReceipt.from_result(result))
        encode(
            RequestStatus(request.identity, result.outcome, acceptance, result=result)
        )

    def _confirm(
        self, request: ComputeRequest, transition: Transition
    ) -> ExecutionReceipt:
        try:
            boundary = self._state.complete(transition)
            receipt = ExecutionReceipt(
                request.identity,
                request.operation,
                "skipped" if transition.skipped else "completed",
                boundary.completed_updates,
                boundary.accumulation,
            )
            self._records.executed(receipt)
            return receipt
        except BaseException:
            self._state.invalidate()
            raise

    def _publish[Work: CompletedWork](
        self,
        request: ComputeRequest,
        receipt: ExecutionReceipt,
        work: Work,
        publish: Callable[[ExecutionReceipt, Work], ComputeResult],
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

    def _failed(self, request: ComputeRequest, failure: Failure) -> RequestStatus:
        try:
            status = self._records.fail(request.identity.request_id, failure)
            self.order.fail_dependents(request.identity.request_id)
            return status
        except BaseException:
            self._state.invalidate()
            raise
