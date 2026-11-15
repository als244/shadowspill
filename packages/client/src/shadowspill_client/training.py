"""Live request submission only; preparation and worker startup belong elsewhere."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import Condition, Lock
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from ._async import CallFuture
from .buffers import BufferBudget
from .datum import Datum
from .plans import PlansHandle
from .protocol.controls import Scalar
from .protocol.payload import MEDIA_TYPE, snapshot_batch
from .protocol.requests import (
    ArtifactRef,
    Operation,
    OperationRequest,
    RequestIdentity,
    WriterLease,
    identifier,
)
from .protocol.results import RequestStatus
from .protocol.runs import RunInfo
from .protocol.training import ResolvedTraining
from .results import OperationFuture, TrainingRequestError, TrainingResult

if TYPE_CHECKING:
    from .service import ServiceClient


class RequestQueries:
    def __init__(self, training: TrainingClient) -> None:
        self._training = training

    def status(self, request_id: str) -> CallFuture[RequestStatus]:
        identifier(request_id, "request ID")
        training = self._training
        return training._service._work.submit(lambda: training._status(request_id))

    def get(self, request_id: str) -> CallFuture[TrainingResult]:
        identifier(request_id, "request ID")
        training = self._training
        return training._service._work.submit(lambda: training._wait(request_id))


class TrainingClient:
    """A handle to ready workers; construction is owned by ServiceClient factories.

    Calls copy complete inputs before returning, then upload in submission order.
    The server alone validates coverage, packs inputs and advances training state.
    A connection failure requires explicit lookup/reattachment, never replay.
    """

    def __init__(
        self,
        service: ServiceClient,
        info: RunInfo,
        config: ResolvedTraining,
        writer: WriterLease | None,
    ) -> None:
        self._service, self._config, self._writer = service, config, writer
        self.session_id, self.run_id = info.session_id, info.run_id
        self.read_only = writer is None
        self.plans = None if info.plans is None else PlansHandle(info.plans, service)
        self.requests = RequestQueries(self)
        self._route = f"/v1/runs/{self.session_id}/{self.run_id}"
        self._submission, self._upload, self._state = Lock(), Condition(), Lock()
        self._next = 1 if writer is None else writer.next_sequence
        self._upload_next = self._next
        self._pending: dict[str, OperationFuture] = {}
        self._blocked = False
        self._snapshots = BufferBudget(
            config.execution.max_staging_bytes, config.execution.max_pending_requests
        )
        self._slots = BufferBudget(1, config.execution.max_pending_requests)
        service._retain_handle(self)

    def get_config(self) -> ResolvedTraining:
        return self._config

    def _disconnect(self) -> None:
        self._slots.close()
        self._snapshots.close()

    def _status(self, request_id: str, *, wait: bool = False) -> RequestStatus:
        path = self._route + "/requests/" + request_id
        if wait:
            path += "?wait=completed&timeout=1"
        status = self._service._request("GET", path, RequestStatus)
        if (
            status.identity.request_id != request_id
            or status.identity.run_id != self.run_id
        ):
            raise ValueError("request lookup crossed run or request identity")
        return status

    def _wait(
        self,
        request_id: str,
        future: OperationFuture | None = None,
        status: RequestStatus | None = None,
    ) -> TrainingResult:
        while True:
            if status is None:
                status = self._status(request_id, wait=True)
            if future is not None:
                future._observe(status)
            if status.failure is not None:
                raise TrainingRequestError(status)
            if status.result is not None:
                return TrainingResult(status.result, self._service)
            self._service._work.check()
            status = None

    def _submit(
        self,
        operation: Operation,
        batch: Sequence[Datum] | None = None,
        *,
        optim: Mapping[str, Scalar] | None = None,
        loss_params: Mapping[str, Scalar] | None = None,
        grad_scale: float | None = None,
        outputs: str | None = None,
        mode: Literal["evaluation", "training"] | None = None,
        runtime_trace: bool = False,
    ) -> OperationFuture:
        self._service._work.check()
        if self._writer is None:
            raise PermissionError("read-only training handle cannot submit operations")
        if operation in ("step", "forward_backward", "forward") and batch is None:
            raise TypeError("global_batch must be a list or tuple of Datum records")
        # Submission serialization includes snapshot admission; completion uses a
        # separate lock and can always release the next caller's pending slot.
        with self._submission:
            with self._state:
                dependencies = tuple(self._pending)
            slot = self._slots.reserve(0)
            snapshot = None
            request_id = uuid4().hex
            queued = False
            try:
                with self._state:
                    if self._blocked:
                        raise ConnectionError(
                            "a submission exchange failed; inspect saved request IDs "
                            "and reattach explicitly before submitting more work"
                        )
                    identity = RequestIdentity(
                        request_id,
                        self.run_id,
                        self._next,
                        self._writer.writer_id,
                        self._writer.writer_epoch,
                        self._writer.server_incarnation,
                    )
                # Freeze controls before potentially blocking input copies.
                request = OperationRequest(
                    identity,
                    operation,
                    optim={} if optim is None else optim,
                    loss_params={} if loss_params is None else loss_params,
                    grad_scale=grad_scale,
                    outputs=outputs,
                    mode=mode,
                    runtime_trace=runtime_trace,
                    dependencies=dependencies,
                )
                if batch is not None:
                    from dataclasses import replace

                    snapshot = snapshot_batch(batch, self._snapshots)
                    request = replace(
                        request,
                        datum_count=len(snapshot.datums),
                        payload=ArtifactRef(
                            request_id, snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
                        ),
                    )
                future = OperationFuture(identity)
                with self._state:
                    self._next += 1
                    self._pending[request_id] = future

                def cleanup() -> None:
                    if snapshot is not None:
                        snapshot.close()
                    with self._state:
                        self._pending.pop(request_id, None)
                    slot.close()

                def execute() -> TrainingResult:
                    # Complete each metadata/upload exchange before starting the
                    # next one. Device execution and later input upload overlap.
                    with self._upload:
                        while identity.sequence != self._upload_next:
                            self._service._work.check()
                            with self._state:
                                if self._blocked:
                                    break
                            self._upload.wait(0.05)
                        with self._state:
                            if self._blocked:
                                raise ConnectionError(
                                    "earlier submission failed; "
                                    "this request was not sent"
                                )
                        try:
                            status = self._service._request(
                                "POST",
                                self._route + "/requests",
                                RequestStatus,
                                request,
                            )
                            future._observe(status)
                            if status.state == "uploading":
                                path = (
                                    self._route + "/requests/" + request_id + "/input"
                                )
                                if snapshot is None:
                                    status = self._service._request(
                                        "PUT", path, RequestStatus
                                    )
                                else:
                                    assert request.payload is not None
                                    value = self._service._transport.upload(
                                        path, request.payload, snapshot.chunks()
                                    ).value
                                    if not isinstance(value, RequestStatus):
                                        raise ValueError(
                                            "upload returned no request status"
                                        )
                                    status = value
                                future._observe(status)
                        except BaseException:
                            with self._state:
                                self._blocked = True
                            raise
                        finally:
                            if snapshot is not None:
                                snapshot.close()
                            self._upload_next += 1
                            self._upload.notify_all()
                    try:
                        return self._wait(request_id, future, status)
                    except TrainingRequestError:
                        raise
                    except BaseException:
                        with self._state:
                            self._blocked = True
                        raise

                self._service._work.submit(execute, future=future, cleanup=cleanup)
                queued = True
                return future
            finally:
                if not queued:
                    if snapshot is not None:
                        snapshot.close()
                    with self._state:
                        if request_id in self._pending:
                            self._blocked = True
                        self._pending.pop(request_id, None)
                    slot.close()

    def step(
        self,
        global_batch: Sequence[Datum],
        *,
        optim: Mapping[str, Scalar] | None = None,
        loss_params: Mapping[str, Scalar] | None = None,
        grad_scale: float = 1,
        outputs: str | None = None,
        runtime_trace: bool = False,
    ) -> OperationFuture:
        return self._submit(
            "step",
            global_batch,
            optim=optim,
            loss_params=loss_params,
            grad_scale=grad_scale,
            outputs=outputs,
            runtime_trace=runtime_trace,
        )

    def forward_backward(
        self,
        global_batch: Sequence[Datum],
        *,
        loss_params: Mapping[str, Scalar] | None = None,
        outputs: str | None = None,
        runtime_trace: bool = False,
    ) -> OperationFuture:
        return self._submit(
            "forward_backward",
            global_batch,
            loss_params=loss_params,
            outputs=outputs,
            runtime_trace=runtime_trace,
        )

    def optim_step(
        self,
        *,
        optim: Mapping[str, Scalar] | None = None,
        grad_scale: float = 1,
        runtime_trace: bool = False,
    ) -> OperationFuture:
        return self._submit(
            "optim_step",
            optim=optim,
            grad_scale=grad_scale,
            runtime_trace=runtime_trace,
        )

    def forward(
        self,
        global_batch: Sequence[Datum],
        *,
        outputs: str | None = None,
        mode: Literal["evaluation", "training"] = "evaluation",
        loss_params: Mapping[str, Scalar] | None = None,
        runtime_trace: bool = False,
    ) -> OperationFuture:
        return self._submit(
            "forward",
            global_batch,
            outputs=outputs,
            mode=mode,
            loss_params=loss_params,
            runtime_trace=runtime_trace,
        )

    def discard_gradients(self) -> OperationFuture:
        return self._submit("discard")

    def close(self) -> OperationFuture:
        return self._submit("close")
