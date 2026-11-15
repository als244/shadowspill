"""Publish prepared caller-order tensors and metrics after confirmed execution."""

import hashlib
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Self

from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill.training.result_frames import ResultFrames
from shadowspill.training.result_metrics import result_metrics
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.worker.requests import WorkerResults
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import MEDIA_TYPE
from shadowspill_client.protocol.requests import ArtifactRef, identifier
from shadowspill_client.protocol.results import (
    ArtifactIssue,
    ExecutionReceipt,
    OperationResult,
)

from .artifacts import ArtifactStore


class ResultPublisher:
    """I/O adapter for RunController; no execution, state transitions or replay."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        session_id: str,
        root: Path,
        memory: BufferBudget,
        disk: BufferBudget,
    ) -> None:
        self.session_id = identifier(session_id, "session ID")
        self.artifacts, self.root, self.memory, self.disk = (
            artifacts,
            root,
            memory,
            disk,
        )
        self._busy = Lock()
        self._owners: set[ResultFrames] = set()
        self.closed = False

    def __call__(
        self, call: PreparedCall, receipt: ExecutionReceipt, work: WorkerResults
    ) -> OperationResult:
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("another result publication is in flight")
        try:
            if self.closed:
                raise RuntimeError("result publisher is closed")
            if (
                receipt.identity != call.request.identity
                or receipt.operation != call.request.operation
            ):
                raise ValueError("publication differs from confirmed execution")
            work.verify()
            loss: float | int | None = None
            metrics: dict[str, float | int] = {}
            outputs: dict[str, ArtifactRef] = {}
            checkpoint = None
            weights = None
            metadata = {}
            if call.request.operation == "save":
                if work.checkpoint is None:
                    raise ValueError("save has no owned checkpoint generation")
                saved = work.checkpoint.commit()[-1]
                assert call.request.save is not None
                if call.request.save.kind == "checkpoint":
                    checkpoint = saved
                else:
                    weights = saved
            if call.request.operation == "restore":
                if call.restore is None:
                    raise ValueError("restore has no accepted checkpoint input")
                checkpoint = call.restore.reference
                metadata = dict(call.restore.manifest.info.metadata)
            if call.key is not None and call.key.operation != "optim_step":
                assert call.candidate is not None
                layout = call.candidate.source.layout.outputs
                if call.assignment is None:
                    raise ValueError(
                        "result publication is missing owned packing metadata"
                    )
                loss, metrics = result_metrics(
                    layout, work.frames, work.open, self.memory
                )
                if layout.outputs.tensors:
                    owner = ResultFrames(
                        layout,
                        call.assignment,
                        root=self.root,
                        memory=self.memory,
                        disk=self.disk,
                    )
                    self._owners.add(owner)
                    try:
                        owner.start()
                        owner.assemble(work.frames, work.open)
                        outputs = self._publish(call, owner)
                    finally:
                        owner.close()
                        self._owners.remove(owner)
            traces, issues = (
                self._traces(work) if call.request.runtime_trace else ((), ())
            )
            return OperationResult(
                call.request.identity,
                call.request.operation,
                receipt.history_id,
                receipt.outcome,
                receipt.completed_updates,
                receipt.accumulation,
                metrics=metrics,
                outputs=outputs,
                optim=call.controls.optim,
                loss_params=call.controls.loss_params,
                grad_scale=call.controls.grad_scale,
                loss=loss,
                state_valid=receipt.state_valid,
                checkpoint=checkpoint,
                weights=weights,
                metadata=metadata,
                traces=traces,
                artifact_issues=issues,
            )
        finally:
            self._busy.release()

    def _traces(
        self, work: WorkerResults
    ) -> tuple[tuple[ArtifactRef, ...], tuple[ArtifactIssue, ...]]:
        references, issues = [], []
        for rank, outcome in work.traces().items():
            if isinstance(outcome, ArtifactIssue):
                issues.append(outcome)
                continue
            try:
                references.append(
                    self.artifacts.put(outcome, partial(work.open_trace, rank))
                )
            except Exception as error:
                message = (str(error).strip() or type(error).__name__)[:1024].rstrip()
                issues.append(
                    ArtifactIssue("runtime_trace_publication", message, outcome.id)
                )
        return tuple(references), tuple(issues)

    def _publish(
        self, call: PreparedCall, owner: ResultFrames
    ) -> dict[str, ArtifactRef]:
        outputs = {}
        identity = call.request.identity
        for name, frame in owner.frames.items():
            key = "\0".join(
                (self.session_id, identity.run_id, identity.request_id, name)
            ).encode()
            artifact_id = "result." + hashlib.sha256(key).hexdigest()
            outputs[name] = self.artifacts.produce(
                OutputSpec(artifact_id, frame.size.wire, MEDIA_TYPE),
                partial(owner.write, name, artifact_id=artifact_id),
            )
        return outputs

    def close(self) -> None:
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("result publication still owns its scratch")
        try:
            for owner in tuple(self._owners):
                owner.close()
                self._owners.remove(owner)
            self.closed = True
        finally:
            self._busy.release()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("result publisher is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
