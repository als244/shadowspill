"""A connection-independent queue around the existing acceptance and dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from threading import Condition, Event, RLock, Thread
from time import monotonic
from typing import Protocol, Self

from shadowspill_client._records import number
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload_io import BufferReader
from shadowspill_client.protocol.requests import ComputeRequest, identifier
from shadowspill_client.protocol.results import (
    ComputeResult,
    ExecutionReceipt,
    RequestStatus,
)

from .dispatch import CompletedWork, DispatchRecords, RequestDispatcher
from .inputs import RequestInputs
from .prepared import PreparedCall, PreparedRequests
from .recovery import InterruptedRequests, fail_interrupted_requests
from .state import Boundary, OperationState, Transition
from .writers import Attachment, WriterOwnership, WriterRecords


class ControllerRecords(DispatchRecords, WriterRecords, InterruptedRequests, Protocol):
    pass


class ControllerWorkers[Work: CompletedWork](Protocol):
    def execute(self, call: PreparedCall, transition: Transition) -> Work: ...

    def close(self) -> None: ...


class RunController[Work: CompletedWork]:
    """Submit/upload/query without keeping the submitting connection alive.

    There is one background consumer and the same OperationState, RequestInputs
    and RequestDispatcher used by direct integration. The publisher runs only
    after the durable execution receipt. Status reads and writer reattachment do
    not execute work. A timeout on a waiter does not cancel an accepted request.

    This is an internal live-run owner. ``close`` requires an idle queue and the
    existing close-state guard; it is not the public ordered close/checkpoint RPC.
    Records remain owned by the service and readable after worker cleanup.
    """

    def __init__(
        self,
        records: ControllerRecords,
        prepared: PreparedRequests,
        workers: ControllerWorkers[Work],
        publish: Callable[[PreparedCall, ExecutionReceipt, Work], ComputeResult],
        *,
        server_incarnation: str,
        budget: BufferBudget,
        state: OperationState | None = None,
    ) -> None:
        identifier(server_incarnation, "server incarnation")
        if records.last_sequence:
            raise ValueError(
                "live controllers require fresh records; recover old runs separately"
            )
        self._records, self._prepared, self._workers = records, prepared, workers
        self._publish, self._incarnation = publish, server_incarnation
        self._state = state if state is not None else OperationState()
        execution = prepared.setup.config.execution
        self._dispatcher = RequestDispatcher(
            records, self._state, max_pending=execution.max_pending_requests
        )
        self.inputs = RequestInputs(
            records, self._dispatcher.order, execution, prepared, budget=budget
        )
        self._writers = WriterOwnership(records, self._dispatcher.order, self.inputs)
        self._wake, self._changed, self._closing_lock = Event(), Condition(), RLock()
        self._active = self._closing = self.closed = False
        self._error: str | None = None
        self._thread = Thread(
            target=self._run, name="shadowspill-controller", daemon=True
        )
        self._thread.start()

    @property
    def boundary(self) -> Boundary:
        with self._dispatcher.order.synchronized():
            return self._state.boundary

    def _usable(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"live controller failed: {self._error}")
        if self._closing or self.closed:
            raise RuntimeError("live controller is closing or closed")

    def _notify(self, *, dispatch: bool = True) -> None:
        if dispatch:
            self._wake.set()
        # Call only outside the acceptance lock: waiters read records while
        # holding this condition to avoid missing a completion notification.
        with self._changed:
            self._changed.notify_all()

    def attach(self, writer_id: str) -> Attachment:
        try:
            with self._dispatcher.order.synchronized():
                self._usable()
                return self._writers.attach(writer_id, self._incarnation)
        finally:
            self._notify()

    def register(self, request: ComputeRequest) -> RequestStatus:
        """Reserve metadata/identity; a new request is not yet accepted."""
        try:
            with self._dispatcher.order.synchronized():
                if self._records.get(request.identity.request_id) is not None:
                    return self._records.submit(request)  # Exact digest, never replay.
                self._usable()
                return self._prepared.register(self.inputs, request)
        finally:
            self._notify()

    def receive(
        self,
        request_id: str,
        stream: BufferReader | None = None,
        *,
        timeout: float | None = 0,
    ) -> RequestStatus:
        """Complete bounded preprocessing; the background consumer owns execution."""
        try:
            with self._dispatcher.order.synchronized():
                status = self.status(request_id)
                if status.state != "uploading":
                    return status
                self._usable()
            return self.inputs.receive(request_id, stream, timeout=timeout)
        finally:
            self._notify()

    def status(self, request_id: str) -> RequestStatus:
        identifier(request_id, "request ID")
        with self._dispatcher.order.synchronized():
            row = self._records.get(request_id)
            if row is None:
                raise KeyError(f"unknown request: {request_id}")
            return row[1]

    def wait(
        self, request_id: str, *, accepted: bool = False, timeout: float | None = None
    ) -> RequestStatus:
        """Wait for recorded acceptance or a terminal result, without owning work."""
        if type(accepted) is not bool:
            raise TypeError("accepted must be boolean")
        deadline = self._deadline(timeout)
        with self._changed:
            while True:
                status = self.status(request_id)
                if status.state in ("completed", "skipped", "failed") or (
                    accepted and status.acceptance is not None
                ):
                    return status
                if self._error is not None:
                    raise RuntimeError(f"live controller failed: {self._error}")
                self._wait(deadline)

    def drain(self, *, timeout: float | None = None) -> None:
        """Wait for existing uploads/accepted work; this does not cancel producers."""
        deadline = self._deadline(timeout)
        with self._changed:
            while True:
                with self._dispatcher.order.synchronized():
                    if self._error is not None:
                        raise RuntimeError(f"live controller failed: {self._error}")
                    if not self._active and not self._records.pending(limit=1):
                        return
                self._wait(deadline)

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        return (
            None
            if timeout is None
            else monotonic() + number(timeout, "wait timeout", minimum=0)
        )

    def _wait(self, deadline: float | None) -> None:
        remaining = None if deadline is None else deadline - monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("waiting for the recorded request boundary")
        self._changed.wait(remaining)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                with self._dispatcher.order.synchronized():
                    if self._closing:
                        return
                try:
                    if not self._one():
                        break
                except BaseException as error:
                    self._failed(error)
                    return
                finally:
                    with self._dispatcher.order.synchronized():
                        self._active = False
                    self._notify(dispatch=False)

    def _one(self) -> bool:
        with self._dispatcher.order.synchronized():
            if self._closing:
                return False
            self._active = True
            pending = self._records.pending(limit=1)
            if not pending or pending[0].state != "accepted":
                self._active = False
                return False
            request_id = pending[0].identity.request_id
            call = self.inputs.take(request_id)
        try:
            self._dispatcher.run(
                request_id,
                lambda transition: self._workers.execute(call, transition),
                lambda receipt, work: self._publish(call, receipt, work),
                participates=call.participates,
            )
        finally:
            self.inputs.finish(request_id)
        with self._dispatcher.order.synchronized():
            writer = self._records.writer
            if writer is not None:
                self._writers.status(writer.writer_id)
        return True

    def _failed(self, error: BaseException) -> None:
        # Retain a bounded diagnostic, never a traceback holding payload buffers.
        with self._dispatcher.order.synchronized():
            self._error = (str(error) or type(error).__name__)[:1024]
            self._state.invalidate()
        with suppress(BaseException):
            self._workers.close()
        with self._dispatcher.order.synchronized():
            for cleanup in (
                self.inputs.abort,
                lambda: fail_interrupted_requests(
                    self._records,
                    self._state,
                    code="controller_failed",
                    message="Controller failed; restore state from a checkpoint.",
                ),
            ):
                # Failed owners remain reachable; broken storage never permits replay.
                with suppress(BaseException):
                    cleanup()

    def close(self) -> None:
        """Close an idle resolved run, or retry cleanup after controller failure."""
        with self._closing_lock:
            with self._dispatcher.order.synchronized():
                if self.closed:
                    return
                if self._active or (
                    self._error is None and self._records.pending(limit=1)
                ):
                    raise RuntimeError(
                        "drain the controller before closing live workers"
                    )
                transition = (
                    None
                    if self._error is not None
                    else self._state.begin("close", "controller-close")
                )
                self._closing = True
                self.inputs.pause_registration(True)
            self._notify()
            self._thread.join()
            try:
                self._workers.close()
                if self._error is not None:
                    self.inputs.abort()
                if self.inputs.staging.used != (0, 0):
                    raise RuntimeError(
                        "input producers still retain controller storage"
                    )
                self.inputs.staging.close()
                with self._dispatcher.order.synchronized():
                    if transition is None:
                        transition = self._state.begin("close", "controller-close")
                    self._state.complete(transition)
                    self.closed = True
            except BaseException:
                with self._dispatcher.order.synchronized():
                    self._state.invalidate()
                raise
            finally:
                self._notify(dispatch=False)

    def __enter__(self) -> Self:
        self._usable()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
