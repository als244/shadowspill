"""Request file ownership around the existing ordered dispatcher/worker group."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock, RLock
from types import MappingProxyType
from typing import BinaryIO, Self

from shadowspill.orchestration.launch.file_grants import (
    FileGrant,
    FileGrants,
    resolve_file,
)
from shadowspill.orchestration.launch.publication import (
    OutputFile,
    OutputFiles,
    OutputWriter,
)
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill.training.session.dispatch import ExecutionNotStarted
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import Transition
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.checkpoints import SaveInfo
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.results import ArtifactIssue

from .checkpoint_records import RankSave, RankSaved, checkpoint_id
from .checkpoint_storage import CheckpointFiles, CheckpointStorage
from .group import WorkerGroup
from .output_files import FileOutputs
from .output_frame import WorkerOutputFrame
from .protocol import WorkerCall, WorkerReply
from .runtime import PreparedWorker
from .trace_files import TraceFiles


@dataclass(frozen=True)
class NodeFiles:
    """Borrow one agent's live file services; its owner closes them after workers."""

    root: Path
    disk: BufferBudget
    memory: BufferBudget
    grants: FileGrants
    outputs: OutputFiles


class WorkerRequests:
    """Local agent composition; state transitions still belong to RequestDispatcher.

    All participants' input delivery and output reservations precede any stage
    command. Confirmed results retain files for the publisher; verification and
    onward transfer happen only after the dispatcher records execution effects.
    """

    def __init__(
        self,
        group: WorkerGroup,
        prepared: Mapping[int, PreparedWorker],
        nodes: Mapping[int, NodeFiles],
        *,
        metadata: BufferBudget,
        checkpoints: CheckpointStorage | None = None,
    ) -> None:
        if (
            group.connections.keys() != prepared.keys()
            or prepared.keys() != nodes.keys()
        ):
            raise ValueError("worker file services require the complete admitted group")
        if any(
            worker.startup.scope != group.connections[rank].scope
            for rank, worker in prepared.items()
        ):
            raise ValueError("worker file services belong to another incarnation")
        self.group = group
        self.prepared, self.nodes = (
            MappingProxyType(dict(prepared)),
            MappingProxyType(dict(nodes)),
        )
        self._metadata = metadata
        self._checkpoints = checkpoints
        # Bound retained command objects plus a full codec envelope per worker
        # before creating them. Encoding itself remains serial and bounded.
        self._bytes = (4096 + 8 * MAX_METADATA_BYTES) * len(prepared)
        self._busy, self._owners_lock = Lock(), RLock()
        self._owners: set[WorkerResults] = set()
        self.closed = False

    def execute(self, call: PreparedCall, transition: Transition) -> WorkerResults:
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("another worker file invocation is in flight")
        try:
            closing = call.request.operation == "close"
            if self.closed:
                raise RuntimeError("worker request delivery is closed")
            if not self.group.valid and not closing:
                self.group.close()
                raise RuntimeError("worker group is invalid; never replay execution")
            cleanup_only = closing and (
                not transition.before.valid or not self.group.valid
            )
            if cleanup_only:
                # Failed channels cannot receive another invocation. Confirm
                # shutdown and retire previous owners before reserving metadata
                # again: a failed stop may still own the full command arena.
                self.group.close()
                self._finish_results()
            try:
                lease = self._metadata.reserve(self._bytes, timeout=0)
            except (ValueError, TimeoutError, RuntimeError) as error:
                raise ExecutionNotStarted(str(error)) from error
            result = WorkerResults(self, lease)
            with self._owners_lock:
                self._owners.add(result)
            if cleanup_only:
                self.closed = result.ready = True
                return result
            try:
                checkpoint = None
                if call.request.operation == "save":
                    if self._checkpoints is None:
                        raise ValueError("run has no checkpoint storage")
                    checkpoint = self._checkpoints.describe(call, transition)
                commands = self.group.commands(call, transition, checkpoint=checkpoint)
                self._prepare(result, call, commands)
            except BaseException as error:
                try:
                    result.close()
                except BaseException as cleanup_error:
                    # Keep the owner and its quotas reachable for explicit retry.
                    if (
                        isinstance(error, Exception)
                        and isinstance(cleanup_error, Exception)
                        and self.group.valid
                    ):
                        raise ExecutionNotStarted(
                            "worker file preparation failed; cleanup retains its quotas"
                        ) from cleanup_error
                    raise
                if isinstance(error, Exception) and self.group.valid:
                    raise ExecutionNotStarted(str(error)) from error
                raise
            try:
                result._begin()
            except BaseException as error:
                result._finished()
                try:
                    result.close()
                except Exception as cleanup_error:
                    if isinstance(error, Exception):
                        raise ExecutionNotStarted(
                            "worker producers were not released; cleanup retains quotas"
                        ) from cleanup_error
                    raise
                if isinstance(error, Exception):
                    raise ExecutionNotStarted(str(error)) from error
                raise
            try:
                replies = (
                    self.group.run(commands)
                    if checkpoint is None
                    else self._save(result, commands, checkpoint)
                )
                if call.restore is not None:
                    assert self._checkpoints is not None
                    self._checkpoints.restored(call.restore)
                if closing:
                    # Executor close replies precede process/log finalization.
                    # The dispatcher's receipt requires confirmed child exit.
                    self.group.close()
                    self.closed = True
            except BaseException as error:
                if isinstance(error, ExecutionNotStarted) or self.group.closed:
                    result._finished()
                    try:
                        result.close()
                    except Exception as cleanup_error:
                        if isinstance(error, ExecutionNotStarted):
                            raise ExecutionNotStarted(
                                "worker staging aborted; cleanup retains its quotas"
                            ) from cleanup_error
                        raise
                # Failed owned-child cleanup leaves active producers and all
                # grants reachable; close() must confirm shutdown before deletion.
                raise
            result._finished()
            result.replies = replies
            result.ready = True
            return result
        finally:
            self._busy.release()

    def _save(
        self,
        result: WorkerResults,
        commands: Mapping[int, WorkerCall],
        info: SaveInfo,
    ) -> Mapping[int, WorkerReply]:
        assert self._checkpoints is not None
        with self.group.staged(commands) as invocation:
            descriptions = []
            for rank, reply in invocation.ready.items():
                description = reply.checkpoint
                if (
                    description is None
                    or description.rank != rank
                    or description.kind != info.kind
                ):
                    raise ValueError("worker omitted its checkpoint admission")
                worker = self.prepared[rank]
                plans = worker.bindings.prepared.plans[rank]
                if description.program != plans[
                    worker.bindings.primary
                ].program.digest or dict(description.additional_programs) != {
                    name: plan.program.digest
                    for name, plan in plans.items()
                    if name != worker.bindings.primary
                }:
                    raise ValueError(
                        "checkpoint admission differs from prepared Programs"
                    )
                descriptions.append(description)

            def write(rank: RankSave, sink: OutputWriter) -> RankSaved:
                if not invocation.released:
                    invocation.execute()
                invocation.connections[rank.rank].receive_stream(
                    sink, size_bytes=rank.size_bytes
                )
                reply = invocation.finish(rank.rank)
                if (
                    reply.saved is None
                    or reply.result
                    != rank.reference(
                        f"{checkpoint_id(info)}.rank.{rank.rank}", reply.saved.sha256
                    ).archive
                ):
                    raise ValueError(
                        "worker omitted its complete checkpoint confirmation"
                    )
                return reply.saved

            settings = next(iter(commands.values())).request.save
            assert settings is not None
            # Install the owner before the group context finalizer can fail.
            # WorkerResults remains registered even if cleanup must be retried.
            try:
                result.checkpoint = self._checkpoints.stage(
                    info, descriptions, write, name=settings.name
                )
            except Exception as error:
                if not invocation.complete:
                    raise
                # Every rank confirmed completion. Failed metadata/file assembly
                # is publication failure, not an uncertain worker invocation.
                result.publication_error = (str(error).strip() or type(error).__name__)[
                    :1024
                ]
            return invocation.results

    def _prepare(
        self,
        result: WorkerResults,
        call: PreparedCall,
        commands: Mapping[int, WorkerCall],
    ) -> None:
        sources = (
            {frame.reference.id: frame for frame in call.spool.frames.values()}
            if call.spool is not None
            else {}
        )
        grouped: dict[NodeFiles, dict[str, ArtifactRef]] = {}
        for rank, command in commands.items():
            references = grouped.setdefault(self.nodes[rank], {})
            for ref in command.frames:
                if references.setdefault(ref.id, ref) != ref:
                    raise ValueError(
                        "node input ID has conflicting artifact references"
                    )
                if ref.id not in sources or sources[ref.id].reference != ref:
                    raise ValueError("worker input reference is not owned by the call")
            if command.restore is not None:
                if (
                    self._checkpoints is None
                    or call.restore is None
                    or command.restore.checkpoint != call.restore.reference
                ):
                    raise ValueError("restore has no accepted checkpoint storage")
                dependencies = [command.restore.rank.archive]
                base = command.restore.info.frozen_base
                if base is not None and base.artifact is not None:
                    dependencies.append(base.artifact)
                for ref in dependencies:
                    if references.setdefault(ref.id, ref) != ref:
                        raise ValueError(
                            "checkpoint dependency ID has conflicting references"
                        )
        for node, references in grouped.items():
            missing = []
            for reference in references.values():
                retained = node.grants.retain(reference)
                if retained is None:
                    missing.append(reference)
                else:
                    result._grants.append(retained)
            if not missing:
                continue
            delivery = FileDelivery(
                missing,
                root=node.root,
                disk=node.disk,
                memory=node.memory,
            )
            result._deliveries.append(delivery)

            def open_source(ref: ArtifactRef) -> AbstractContextManager[BinaryIO]:
                if ref.id in sources:
                    return sources[ref.id].path.open("rb")
                assert self._checkpoints is not None
                return self._checkpoints.open(ref)

            delivery.receive(open_source)
            result._grants.append(node.grants.lend(delivery, missing))
        for rank, command in commands.items():
            node, worker = self.nodes[rank], self.prepared[rank]
            bound = worker.bindings.bind(
                command, partial(resolve_file, node.grants.root)
            )
            sink = FileOutputs(worker.outputs, root=node.outputs.root)
            spec = sink.spec(bound)
            if spec is not None:
                result._files[rank] = node.outputs.reserve(spec)
                assert bound.plan is not None and command.candidate is not None
                result.frames[rank] = worker.outputs[command.candidate, bound.plan]
            if command.request.runtime_trace:
                trace = TraceFiles(
                    worker.traces, bundle=worker.startup.bundle, root=node.outputs.root
                ).spec(bound)
                if trace is not None:
                    result._traces[rank] = node.outputs.reserve(trace)

    def close(self) -> None:
        # Wake any blocked group I/O before waiting for the request owner. A
        # failed stop remains retryable and cannot authorize grant deletion.
        self.group.close()
        with self._busy:
            self.closed = True
            self._finish_results()

    def _finish_results(self) -> None:
        with self._owners_lock:
            owners = tuple(self._owners)
        for result in owners:
            result._finished()
            result.close()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("worker request delivery is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class WorkerResults:
    """CompletedWork for the sole dispatcher; never verify before its receipt."""

    def __init__(self, owner: WorkerRequests, metadata: BufferLease) -> None:
        self._owner, self._metadata = owner, metadata
        self._deliveries: list[FileDelivery] = []
        self._grants: list[FileGrant] = []
        self._files: dict[int, OutputFile] = {}
        self._traces: dict[int, OutputFile] = {}
        self.frames: dict[int, WorkerOutputFrame] = {}
        self.replies: Mapping[int, WorkerReply] = {}
        self.checkpoint: CheckpointFiles | None = None
        self.publication_error: str | None = None
        self._active = self.ready = self.closed = False
        self._producing: set[OutputFile] = set()
        self._lock = RLock()

    def _begin(self) -> None:
        with self._lock:
            self._active = True
            for file in (*self._files.values(), *self._traces.values()):
                file.begin()
                self._producing.add(file)

    def _finished(self) -> None:
        with self._lock:
            if self._active:
                for file in tuple(self._producing):
                    file.finished()
                    self._producing.remove(file)
                self._active = False

    def verify(self) -> None:
        """Publisher boundary: disk/worker-publication failures preserve effects."""
        with self._lock:
            if self.closed or not self.ready:
                raise RuntimeError("worker outputs have no confirmed completion")
            if self.publication_error is not None:
                raise OSError(self.publication_error)
            for rank, reply in self.replies.items():
                if reply.status != "completed":
                    raise OSError(reply.message or "worker output publication failed")
                file = self._files.get(rank)
                if file is None:
                    if self.checkpoint is not None:
                        if (
                            reply.saved is None
                            or reply.result not in self.checkpoint.references
                        ):
                            raise ValueError(
                                "checkpoint completion differs from owned files"
                            )
                    elif reply.result is not None:
                        raise ValueError("skipped worker returned an unexpected output")
                else:
                    if reply.result is None:
                        raise ValueError("worker omitted its prepared output reference")
                    file.verify(reply.result)

    @contextmanager
    def open_trace(self, rank: int, reference: ArtifactRef) -> Iterator[BinaryIO]:
        """Verify optional evidence only after execution completion is recorded."""
        with self._lock:
            if self.closed or not self.ready:
                raise ValueError("trace has no completed owner")
            file = self._traces[rank]
            file.verify(reference)
            borrowed = file.open(reference)
            stream = borrowed.__enter__()
        try:
            yield stream
        finally:
            borrowed.__exit__(None, None, None)

    def traces(self) -> Mapping[int, ArtifactRef | ArtifactIssue]:
        with self._lock:
            if self.closed or not self.ready:
                raise ValueError("trace has no completed owner")
            outcomes: dict[int, ArtifactRef | ArtifactIssue] = {}
            for rank, reply in self.replies.items():
                file = self._traces.get(rank)
                if reply.trace_issue is not None:
                    outcomes[rank] = reply.trace_issue
                elif reply.trace is not None:
                    if file is None:
                        outcomes[rank] = ArtifactIssue(
                            "runtime_trace_unreserved",
                            "Worker returned an unreserved trace.",
                        )
                    else:
                        outcomes[rank] = reply.trace
                elif file is not None:
                    outcomes[rank] = ArtifactIssue(
                        "runtime_trace_missing",
                        "Worker omitted its trace evidence.",
                        file.spec.id,
                    )
            return MappingProxyType(outcomes)

    @contextmanager
    def open(self, rank: int) -> Iterator[BinaryIO]:
        """Borrow a verified output; callers must finish reads before work.close()."""
        with self._lock:
            if self.closed or not self.ready or rank not in self._files:
                raise ValueError("worker has no completed output file")
            file, reference = self._files[rank], self.replies[rank].result
            if reference is None:
                raise ValueError("worker has no output reference")
            borrowed = file.open(reference)
            stream = borrowed.__enter__()
        try:
            yield stream
        finally:
            borrowed.__exit__(None, None, None)

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            if self._active:
                raise RuntimeError("worker producers have not confirmed termination")
            if self.checkpoint is not None:
                self.checkpoint.close()
                self.checkpoint = None
            while self._grants:
                self._grants[-1].close()
                self._grants.pop()
            while self._deliveries:
                self._deliveries[-1].close()
                self._deliveries.pop()
            for rank in tuple(self._files):
                self._files[rank].close()
                del self._files[rank]
            for rank in tuple(self._traces):
                self._traces[rank].close()
                del self._traces[rank]
            self.frames.clear()
            self.replies = {}
            self._metadata.close()
            with self._owner._owners_lock:
                self._owner._owners.remove(self)
            self.closed = True
