"""Request file ownership around the existing ordered dispatcher/worker group."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
from shadowspill.orchestration.launch.publication import OutputFile, OutputFiles
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill.training.session.dispatch import ExecutionNotStarted
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import Transition
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES
from shadowspill_client.protocol.requests import ArtifactRef

from .group import WorkerGroup
from .output_files import FileOutputs
from .output_frame import WorkerOutputFrame
from .protocol import WorkerCall, WorkerReply
from .runtime import PreparedWorker


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
            if self.closed:
                raise RuntimeError("worker request delivery is closed")
            if not self.group.valid:
                self.group.close()
                raise RuntimeError("worker group is invalid; never replay execution")
            try:
                lease = self._metadata.reserve(self._bytes, timeout=0)
            except (ValueError, TimeoutError, RuntimeError) as error:
                raise ExecutionNotStarted(str(error)) from error
            result = WorkerResults(self, lease)
            with self._owners_lock:
                self._owners.add(result)
            try:
                commands = self.group.commands(call, transition)
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
                replies = self.group.run(commands)
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
        for node, references in grouped.items():
            if not references:
                continue
            delivery = FileDelivery(
                tuple(references.values()),
                root=node.root,
                disk=node.disk,
                memory=node.memory,
            )
            result._deliveries.append(delivery)
            delivery.receive(lambda ref: sources[ref.id].path.open("rb"))
            result._grants.append(
                node.grants.lend(delivery, tuple(references.values()))
            )
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

    def close(self) -> None:
        # Wake any blocked group I/O before waiting for the request owner. A
        # failed stop remains retryable and cannot authorize grant deletion.
        self.group.close()
        with self._busy:
            self.closed = True
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
        self.frames: dict[int, WorkerOutputFrame] = {}
        self.replies: Mapping[int, WorkerReply] = {}
        self._active = self.ready = self.closed = False
        self._producing: set[int] = set()
        self._lock = RLock()

    def _begin(self) -> None:
        with self._lock:
            self._active = True
            for rank, file in self._files.items():
                file.begin()
                self._producing.add(rank)

    def _finished(self) -> None:
        with self._lock:
            if self._active:
                for rank in tuple(self._producing):
                    self._files[rank].finished()
                    self._producing.remove(rank)
                self._active = False

    def verify(self) -> None:
        """Publisher boundary: disk/worker-publication failures preserve effects."""
        with self._lock:
            if self.closed or not self.ready:
                raise RuntimeError("worker outputs have no confirmed completion")
            for rank, reply in self.replies.items():
                if reply.status != "completed":
                    raise OSError(reply.message or "worker output publication failed")
                file = self._files.get(rank)
                if file is None:
                    if reply.result is not None:
                        raise ValueError("skipped worker returned an unexpected output")
                else:
                    if reply.result is None:
                        raise ValueError("worker omitted its prepared output reference")
                    file.verify(reply.result)

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
            while self._grants:
                self._grants[-1].close()
                self._grants.pop()
            while self._deliveries:
                self._deliveries[-1].close()
                self._deliveries.pop()
            for rank in tuple(self._files):
                self._files[rank].close()
                del self._files[rank]
            self.frames.clear()
            self.replies = {}
            self._metadata.close()
            with self._owner._owners_lock:
                self._owner._owners.remove(self)
            self.closed = True
