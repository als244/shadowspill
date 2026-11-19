"""Live request submission only; preparation and worker startup belong elsewhere."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import Condition, Lock
from time import monotonic
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from ._async import CallFuture
from ._records import Value, number
from .batch_validation import BatchValidation, BatchValidator
from .buffers import BufferBudget
from .captures import NsightCapture
from .checkpoints import CheckpointHandle
from .datum import Datum
from .metrics import MetricQueries
from .plans import PlansHandle
from .protocol.checkpoint_settings import SaveSettings
from .protocol.controls import Scalar
from .protocol.metrics import MetricRecord
from .protocol.payload import MEDIA_TYPE, BatchSnapshot
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
from .reports import ReportHandle, report
from .results import OperationFuture, TrainingRequestError, TrainingResult
from .transport.http import RemoteError, TransportError
from .weights import WeightsHandle

if TYPE_CHECKING:
    from .service import ServiceClient


class RequestQueries:
    """Lookups of a run's requests by ID, available as `TrainingClient.requests`.

    They never resubmit a request, and work on a read-only handle.
    """

    def __init__(self, training: TrainingClient) -> None:
        self._training = training

    def status(self, request_id: str) -> CallFuture[RequestStatus]:
        """Read a request's current state.

        Parameters
        ----------
        request_id : str
            ID of a request in this run, such as an `OperationFuture`'s
            ``request_id``.

        Returns
        -------
        CallFuture of RequestStatus
            The request's state when the service answers. It fails with
            `RemoteError` if the run has no request with this ID.

        Raises
        ------
        ValueError
            If ``request_id`` is not a valid request ID.
        """
        identifier(request_id, "request ID")
        training = self._training
        return training._service._work.submit(lambda: training._status(request_id))

    def get(
        self, request_id: str, *, timeout: float | None = None
    ) -> CallFuture[TrainingResult]:
        """Wait for a request to finish and return its result.

        Parameters
        ----------
        request_id : str
            ID of a request in this run, such as an `OperationFuture`'s
            ``request_id``.
        timeout : float, optional
            Longest wait, in seconds, counted from this call. Past it the
            future fails with `TimeoutError` and stops polling; the request
            goes on and can be looked up again. By default the wait lasts
            until the request finishes.

        Returns
        -------
        CallFuture of TrainingResult
            The request's outcome. It fails with `TrainingRequestError` if the
            request failed.

        Raises
        ------
        ValueError
            If ``request_id`` is not a valid request ID, or ``timeout`` is
            negative.
        """
        identifier(request_id, "request ID")
        deadline = (
            None
            if timeout is None
            else monotonic() + number(timeout, "timeout", minimum=0)
        )
        training = self._training
        return training._service._work.submit(
            lambda: training._wait(request_id, deadline=deadline)
        )


class TrainingClient:
    """A handle to one training run, for submitting operations and reading results.

    `ServiceClient.create_training_client` and
    `ServiceClient.get_training_client` return it, through a `TrainingFuture`.
    Each operation call copies its batch and controls, checks the batch
    locally against the prepared layouts, and returns an `OperationFuture`
    before uploading. The service runs a handle's operations in submission
    order, and checks them again. Each operation depends on the handle's
    operations still unfinished when it was submitted, and fails without
    running if one of them fails. A call blocks while
    ``Execution.max_pending_requests`` operations are unfinished, or while
    batch copies not yet uploaded fill ``Execution.max_staging_bytes``.

    After an exchange with the service fails, the handle refuses further
    operations and nothing is resubmitted: look up the saved request IDs and
    reattach with `ServiceClient.get_training_client`. Attaching a new
    writable handle fences this one, whose later operations then fail.

    Attributes
    ----------
    session_id : str
        The session the run belongs to.
    run_id : str
        The run's ID.
    read_only : bool
        True when the handle cannot submit operations.
    plans : PlansHandle or None
        The run's saved Plans; None when the run records none.
    requests : RequestQueries
        Lookups of the run's requests by ID.
    metrics : MetricQueries
        Queries of the run's recorded metrics.
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
        self.plans = (
            None
            if info.plans is None
            else PlansHandle(info.plans, service, info.run_id)
        )
        self.requests = RequestQueries(self)
        self.metrics = MetricQueries(service, self.session_id, self.run_id)
        self._route = f"/v1/runs/{self.session_id}/{self.run_id}"
        self._submission, self._upload, self._state = Lock(), Condition(), Lock()
        self._next = 1 if writer is None else writer.next_sequence
        self._upload_next = self._next
        self._pending: dict[str, OperationFuture[TrainingResult]] = {}
        self._blocked = False
        self._snapshots = BufferBudget(
            config.execution.max_staging_bytes, config.execution.max_pending_requests
        )
        self._slots = BufferBudget(1, config.execution.max_pending_requests)
        self._validator = BatchValidator(service, config, info.plans)
        service._retain_handle(self)

    def get_config(self) -> ResolvedTraining:
        """Return the run's resolved configuration, without contacting the service.

        Returns
        -------
        ResolvedTraining
            The configuration the run was created with; pass it as
            ``config=`` to create a run with the same settings.
        """
        return self._config

    def validate_batch(
        self,
        batch: Sequence[Datum],
        *,
        operation: str = "step",
        outputs: str | None = None,
        mode: str | None = None,
    ) -> CallFuture[BatchValidation]:
        """Check a batch locally against the prepared layouts, without submitting it.

        The checks are those `step`, `forward_backward` and `forward` make
        before uploading: input values, batch limits, and packing into a
        prepared layout, padding included. Packing is not checked for a model
        that groups its inputs on the service.

        Parameters
        ----------
        batch : sequence of Datum
            The batch, as a list or tuple.
        operation : str, default "step"
            ``"step"``, ``"forward_backward"`` or ``"forward"``.
        outputs : str, optional
            Output set to check for; by default the first one prepared for
            ``operation`` and ``mode``.
        mode : str, optional
            ``"training"`` or ``"evaluation"``; by default ``"evaluation"``
            for ``"forward"`` and ``"training"`` otherwise.

        Returns
        -------
        CallFuture of BatchValidation
            The capacities the batch uses and the prepared layout it fits. It
            fails with `ValueError` if the batch fails a check.

        Raises
        ------
        TypeError
            If ``batch`` is not a list or tuple of `Datum`.
        ValueError
            If ``operation``, ``outputs`` and ``mode`` name nothing prepared,
            or the batch's copy needs more than ``Execution.max_staging_bytes``.
        """
        self._service._work.check()
        snapshot = self._validator.snapshot(
            batch, self._snapshots, operation, outputs, mode
        )
        try:
            return self._service._work.submit(
                lambda: self._validator.validate(snapshot, operation, outputs, mode),
                cleanup=snapshot.close,
            )
        except BaseException:
            snapshot.close()
            raise

    def log_metrics(
        self,
        values: Mapping[str, float | int],
        *,
        request_id: str,
        metric_id: str | None = None,
    ) -> CallFuture[MetricRecord]:
        """Record caller values, such as evaluation scores, for a finished request.

        The record keeps that request's history and update count, and is
        stored apart from the engine's own metrics. It takes no place in the
        training order and works on a read-only handle.

        Parameters
        ----------
        values : mapping of str to float
            One to 128 finite numbers, by name; each name has at most 256
            characters.
        request_id : str
            ID of a request in this run that has completed, been skipped or
            failed.
        metric_id : str, optional
            ID for this record within the request; reusing it with different
            values is refused. By default it is derived from ``values``, so
            repeating an identical call, as after a lost reply, records
            nothing new.

        Returns
        -------
        CallFuture of MetricRecord
            The stored record.

        Raises
        ------
        ValueError
            If ``values``, ``request_id`` or ``metric_id`` is invalid.
        """
        return self.metrics._log(values, request_id=request_id, metric_id=metric_id)

    def nsys(self, *, rank: int = 0) -> CallFuture[NsightCapture]:
        """Read one worker's Nsight Systems capture, for a run created with one.

        The capture is configured with ``Execution(nsys=...)``. Its report is
        finalized after the worker exits, so it can stay pending until `close`
        has finished.

        Parameters
        ----------
        rank : int, default 0
            The worker's rank.

        Returns
        -------
        CallFuture of NsightCapture
            The capture's status and, when present, its report.

        Raises
        ------
        ValueError
            If ``rank`` is negative or not below 65536.
        """
        return self._service.get_nsys(self.run_id, rank=rank)

    def report(
        self,
        *,
        format: Literal["html", "json"] = "html",
        after: int = 0,
        through: int | None = None,
        max_records: int = 10000,
        max_bytes: int = 64 << 20,
    ) -> CallFuture[ReportHandle]:
        """Create a stored report of the run's recorded results.

        The report covers losses, metrics, request outcomes and caller values
        recorded after journal position ``after``, up to and including
        ``through``. Making it runs no training and works after the workers
        have closed. A report that would exceed a limit fails, and nothing is
        stored.

        Parameters
        ----------
        format : str, default "html"
            ``"html"`` or ``"json"``.
        after : int, default 0
            Journal position after which records are included.
        through : int, optional
            Last journal position included; by default the latest one when the
            service makes the report.
        max_records : int, default 10000
            Most records the report may hold, up to 100000.
        max_bytes : int, default 67108864
            Largest report size, in bytes (64 MiB by default), up to 1 GiB.

        Returns
        -------
        CallFuture of ReportHandle
            The stored report, for download.

        Raises
        ------
        ValueError
            If ``format`` is unknown, ``through`` is before ``after``, or a
            limit is out of range.
        """
        return report(
            self._service,
            self.session_id,
            self.run_id,
            format=format,
            after=after,
            through=through,
            max_records=max_records,
            max_bytes=max_bytes,
        )

    def _disconnect(self) -> None:
        self._slots.close()
        self._snapshots.close()

    def _status(
        self, request_id: str, *, wait: bool = False, seconds: float = 1.0
    ) -> RequestStatus:
        path = self._route + "/requests/" + request_id
        if wait:
            path += f"?wait=completed&timeout={seconds:.3f}"
        status = self._service._request("GET", path, RequestStatus)
        if (
            status.identity.request_id != request_id
            or status.identity.run_id != self.run_id
        ):
            raise ValueError("request lookup crossed run or request identity")
        return status

    def _send_input(
        self, request: OperationRequest, snapshot: BatchSnapshot | None
    ) -> RequestStatus:
        """Send one request's input, again while the service is only busy.

        Earlier requests can hold the staging this input needs. The service
        then refuses the upload as busy, or closes the connection before
        reading it; neither changes the request, which still waits for its
        input, so the same input is sent again after a pause. Any other answer
        is the request's own status or a failure. Called holding ``_upload``.
        """
        request_id = request.identity.request_id
        path = self._route + "/requests/" + request_id + "/input"
        delay = 0.1
        while True:
            try:
                if snapshot is None:
                    return self._service._request("PUT", path, RequestStatus)
                assert request.payload is not None
                value = self._service._transport.upload(
                    path, request.payload, snapshot.chunks()
                ).value
                if not isinstance(value, RequestStatus):
                    raise ValueError("upload returned no request status")
                return value
            except (RemoteError, TransportError) as error:
                if isinstance(error, RemoteError) and error.code != "backpressure":
                    raise
                status = self._status(request_id)
                if status.state != "uploading":
                    return status
            self._service._work.check()
            self._upload.wait(delay)
            delay = min(2.0, delay * 2)

    def _wait(
        self,
        request_id: str,
        future: OperationFuture[TrainingResult] | None = None,
        status: RequestStatus | None = None,
        *,
        deadline: float | None = None,
    ) -> TrainingResult:
        while True:
            if status is None:
                seconds = 1.0
                if deadline is not None:
                    seconds = min(seconds, deadline - monotonic())
                    if seconds <= 0:
                        raise TimeoutError(
                            f"request {request_id} has not finished within the "
                            "wait; it continues on the service"
                        )
                status = self._status(request_id, wait=True, seconds=seconds)
            if future is not None:
                future._observe(status)
            if status.failure is not None:
                raise TrainingRequestError(status)
            if status.result is not None:
                if status.result.operation == "close":
                    self._wait_retired()
                return TrainingResult(status.result, self._service)
            self._service._work.check()
            status = None

    def _wait_retired(self) -> None:
        # The durable close receipt can precede worker joins, exit-log
        # publication and the final creation-envelope release. Only observe
        # lifecycle progress here; never resubmit close or retry creation.
        while True:
            self._service._work.check()
            reply = self._service._transport.request("GET", self._route)
            info = reply.value
            if not isinstance(info, RunInfo) or (
                info.owner_id != self._service.info.owner_id
                or info.session_id != self.session_id
                or info.run_id != self.run_id
            ):
                raise ValueError("close lookup crossed authenticated run scope")
            if reply.status == 200 and info.state in ("closed", "failed", "stopped"):
                return
            if reply.status not in (200, 202):
                raise ValueError("unknown run retirement progress")
            self._service._work.pause()

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
        save: SaveSettings | None = None,
        restore: str | None = None,
    ) -> OperationFuture[TrainingResult]:
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
                    save=save,
                    restore=restore,
                )
                if batch is not None:
                    from dataclasses import replace

                    snapshot = self._validator.snapshot(
                        batch, self._snapshots, operation, outputs, mode
                    )
                    self._validator.validate(snapshot, operation, outputs, mode)
                    request = replace(
                        request,
                        datum_count=len(snapshot.datums),
                        payload=ArtifactRef(
                            request_id, snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
                        ),
                    )
                future = OperationFuture[TrainingResult](identity)
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
                                status = self._send_input(request, snapshot)
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
    ) -> OperationFuture[TrainingResult]:
        """Submit one complete update: forward, backward and an optimizer step.

        No gradient accumulation may be open; finish one first with
        `optim_step` or `discard_gradients`. A batch in which no position
        takes part in the loss is rejected.

        Parameters
        ----------
        global_batch : sequence of Datum
            The whole batch for all workers, as a list or tuple. It is copied
            before the call returns, so the caller may change it afterwards.
        optim : mapping of str to scalar, optional
            Optimizer controls for this update, such as ``{"lr": 3e-4}``.
            Controls left out take the run's defaults; no value carries over
            to later calls.
        loss_params : mapping of str to scalar, optional
            Objective controls for this call, resolved the same way.
        grad_scale : float, default 1
            Factor applied to every gradient before the update, such as
            ``1 / len(global_batch)`` to average a loss summed over records.
        outputs : str, optional
            Output set to produce; by default the first one prepared for
            ``"step"``.
        runtime_trace : bool, default False
            Record a runtime trace of this call, one per worker, in the
            result's ``traces``.

        Returns
        -------
        OperationFuture of TrainingResult
            The outcome, with its ``loss``, ``metrics`` and the new
            ``completed_updates``. It fails with `TrainingRequestError` if the
            service rejects or fails the request, as for an unknown control or
            an open accumulation.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        TypeError
            If ``global_batch`` is not a list or tuple of `Datum`, or
            ``optim`` or ``loss_params`` is not a mapping of plain Python
            values.
        ValueError
            If the batch fails a local check or its copy needs more than
            ``Execution.max_staging_bytes``, a control value is neither a bool
            nor a finite number, ``grad_scale`` is not a finite number, or
            ``"step"`` was not prepared with ``outputs``.
        """
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
    ) -> OperationFuture[TrainingResult]:
        """Submit a forward and backward pass that adds to the gradient accumulation.

        The first call opens an accumulation and later calls add to it, until
        `optim_step` applies it or `discard_gradients` drops it. Gradients add
        up unscaled; scale the update once with ``grad_scale`` on
        `optim_step`. A batch in which no position takes part in the loss is
        rejected.

        Parameters
        ----------
        global_batch : sequence of Datum
            The part of the update's batch to add, for all workers, as a list
            or tuple. It is copied before the call returns.
        loss_params : mapping of str to scalar, optional
            Objective controls for this call, as in `step`.
        outputs : str, optional
            Output set to produce; by default the first one prepared for
            ``"forward_backward"``.
        runtime_trace : bool, default False
            Record a runtime trace of this call, as in `step`.

        Returns
        -------
        OperationFuture of TrainingResult
            The outcome, with this part's ``loss`` and ``metrics`` and the
            accumulation's ``contributions`` so far. It fails with
            `TrainingRequestError` if the service rejects or fails the
            request.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        TypeError
            If ``global_batch`` is not a list or tuple of `Datum`, or
            ``loss_params`` is not a mapping of plain Python values.
        ValueError
            If the batch fails a local check or its copy needs more than
            ``Execution.max_staging_bytes``, a control value is neither a bool
            nor a finite number, or ``"forward_backward"`` was not prepared
            with ``outputs``.
        """
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
    ) -> OperationFuture[TrainingResult]:
        """Submit one optimizer update from the accumulated gradients.

        It requires an accumulation opened by `forward_backward`, and closes
        it.

        Parameters
        ----------
        optim : mapping of str to scalar, optional
            Optimizer controls for this update, as in `step`.
        grad_scale : float, default 1
            Factor applied to every accumulated gradient before the update,
            such as ``1 / n`` to average a loss summed over ``n`` records in
            all the parts.
        runtime_trace : bool, default False
            Record a runtime trace of this call, as in `step`.

        Returns
        -------
        OperationFuture of TrainingResult
            The outcome, with the new ``completed_updates``; it has no loss.
            It fails with `TrainingRequestError` if the service rejects or
            fails the request, as when no accumulation is open.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        TypeError
            If ``optim`` is not a mapping of plain Python values.
        ValueError
            If a control value is neither a bool nor a finite number, or
            ``grad_scale`` is not a finite number.
        """
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
    ) -> OperationFuture[TrainingResult]:
        """Submit a forward pass, with no backward pass or update.

        It can run while a gradient accumulation is open.

        Parameters
        ----------
        global_batch : sequence of Datum
            The whole batch for all workers, as a list or tuple. It is copied
            before the call returns.
        outputs : str, optional
            Output set to produce, such as ``"predictions"``; by default the
            first one prepared for ``"forward"`` in ``mode``.
        mode : str, default "evaluation"
            ``"evaluation"`` leaves training state unchanged; ``"training"``
            runs the model as prepared for training. The mode must be
            prepared with ``outputs``.
        loss_params : mapping of str to scalar, optional
            Objective controls for this call, as in `step`.
        runtime_trace : bool, default False
            Record a runtime trace of this call, as in `step`.

        Returns
        -------
        OperationFuture of TrainingResult
            The outcome: ``loss`` and ``metrics`` when the output set computes
            them, and tensor ``outputs`` as handles. It fails with
            `TrainingRequestError` if the service rejects or fails the
            request.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        TypeError
            If ``global_batch`` is not a list or tuple of `Datum`, or
            ``loss_params`` is not a mapping of plain Python values.
        ValueError
            If the batch fails a local check or its copy needs more than
            ``Execution.max_staging_bytes``, a control value is neither a bool
            nor a finite number, or ``"forward"`` was not prepared with
            ``outputs`` and ``mode``.
        """
        return self._submit(
            "forward",
            global_batch,
            outputs=outputs,
            mode=mode,
            loss_params=loss_params,
            runtime_trace=runtime_trace,
        )

    def discard_gradients(self) -> OperationFuture[TrainingResult]:
        """Submit dropping the open gradient accumulation, if any, without an update.

        Returns
        -------
        OperationFuture of TrainingResult
            The outcome; ``completed_updates`` is unchanged. It fails with
            `TrainingRequestError` if the service rejects or fails the
            request.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        """
        return self._submit("discard")

    def save_checkpoint(
        self,
        *,
        name: str | None = None,
        metadata: Mapping[str, Value] | None = None,
        weights: Literal["master", "compute"] = "master",
    ) -> OperationFuture[CheckpointHandle]:
        """Submit saving a checkpoint of the training state in service storage.

        The save runs in the training order, after the operations submitted
        before it, and does not count as an update. No gradient accumulation
        may be open. Nothing is downloaded until the checkpoint's ``download``
        is called.

        Parameters
        ----------
        name : str, optional
            Name for the checkpoint within this run, at most 256 characters,
            usable with `load_checkpoint`. Saving under a name again moves it
            to the new checkpoint; the older one stays.
        metadata : mapping of str to object, optional
            JSON values saved with the checkpoint and returned when it is
            loaded, such as a data position. They are copied at the call.
        weights : str, default "master"
            ``"master"`` saves the master weights; ``"compute"`` saves the
            weights at compute precision.

        Returns
        -------
        OperationFuture of CheckpointHandle
            The saved checkpoint. It fails with `TrainingRequestError` if the
            service rejects or fails the request, as when an accumulation is
            open.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        TypeError
            If ``metadata`` is not a mapping of JSON values.
        ValueError
            If ``name`` is empty, has surrounding spaces or is too long,
            ``metadata`` holds a number that is not finite or a key that is not
            a nonempty string, or ``weights`` is unknown.
        """
        settings = SaveSettings(name, {} if metadata is None else metadata, weights)

        def checkpoint(result: TrainingResult) -> CheckpointHandle:
            if result.checkpoint is None:
                raise ValueError("save completed without a checkpoint reference")
            return result.checkpoint

        return self._submit("save", save=settings)._mapped(checkpoint)

    def load_checkpoint(self, checkpoint_id: str) -> OperationFuture[TrainingResult]:
        """Submit loading a checkpoint into the run's training state.

        The load runs in the training order. No gradient accumulation may be
        open. A name is pinned to one checkpoint when the service accepts the
        request, as `OperationFuture.accepted` reports; moving the name later
        does not change it. The run continues at the saved update count, in a
        new history.

        Parameters
        ----------
        checkpoint_id : str
            A checkpoint's ID, such as a `CheckpointHandle`'s ``id``, or a
            name it was given by `save_checkpoint` in this run.

        Returns
        -------
        OperationFuture of TrainingResult
            The outcome: ``metadata`` holds the caller metadata saved with the
            checkpoint, ``checkpoint`` the loaded checkpoint, and
            ``completed_updates`` the saved count. It fails with
            `TrainingRequestError` if the service rejects or fails the
            request.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        ValueError
            If ``checkpoint_id`` is empty, has surrounding spaces or is longer
            than 256 characters.
        """
        return self._submit("restore", restore=checkpoint_id)

    def export_weights(
        self,
        *,
        name: str | None = None,
        format: str = "shadowspill",
        weights: Literal["master", "compute"] = "master",
        metadata: Mapping[str, Value] | None = None,
    ) -> OperationFuture[WeightsHandle]:
        """Submit exporting the model weights to service storage.

        The export runs in the training order, like `save_checkpoint`, and
        holds the model weights and persistent buffers only: no optimizer
        state or gradients. No gradient accumulation may be open. Nothing is
        downloaded until the export's ``download`` is called.

        Parameters
        ----------
        name : str, optional
            Name for the export within this run, at most 256 characters.
        format : str, default "shadowspill"
            Export format. The service supports ``"shadowspill"`` and fails
            any other before running.
        weights : str, default "master"
            ``"master"`` or ``"compute"``, as in `save_checkpoint`.
        metadata : mapping of str to object, optional
            JSON values saved with the export, as in `save_checkpoint`.

        Returns
        -------
        OperationFuture of WeightsHandle
            The exported weights. It fails with `TrainingRequestError` if the
            service rejects or fails the request.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        TypeError
            If ``metadata`` is not a mapping of JSON values.
        ValueError
            If ``name`` or ``metadata`` is invalid as in `save_checkpoint`,
            ``format`` is empty or longer than 128 characters, or ``weights``
            is unknown.
        """
        settings = SaveSettings(
            name, {} if metadata is None else metadata, weights, "weights", format
        )

        def exported(result: TrainingResult) -> WeightsHandle:
            if result.weights is None:
                raise ValueError("export completed without a weights reference")
            return result.weights

        return self._submit("save", save=settings)._mapped(exported)

    def close(self) -> OperationFuture[TrainingResult]:
        """Submit closing the run, which releases its workers and resources.

        The close runs in the training order, after the operations submitted
        before it; operations submitted after it fail. No gradient
        accumulation may be open, and pending gradients are never applied.
        Results, metrics, reports and saved artifacts stay readable afterwards.

        Returns
        -------
        OperationFuture of TrainingResult
            The outcome, once the run's resources are released and a new run
            can use them. The request's status can show it completed earlier.
            It fails with `TrainingRequestError` if the service rejects the
            close, and with `RemoteError` if releasing the resources fails.

        Raises
        ------
        PermissionError
            If the handle is read-only.
        ConnectionError
            If an earlier exchange on this handle failed, or the connection is
            closed.
        """
        return self._submit("close")
