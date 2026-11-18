"""A sequential worker command loop, with no duplicate training state machine."""

from collections.abc import Callable
from typing import BinaryIO

from shadowspill_client._records import Record
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import BoundWorkerCall
from .channel import read_message, write_message
from .checkpoint_records import RankSaved, checkpoint_id
from .protocol import (
    CODEC,
    WorkerArtifacts,
    WorkerCall,
    WorkerControl,
    WorkerOutputError,
    WorkerReply,
    WorkerScope,
    WorkerTicket,
    ticket,
)


def _message(error: Exception) -> str:
    return (str(error).strip() or type(error).__name__)[:1024].rstrip()


class WorkerLoop:
    """One outstanding command, consumed once before entering the Executor.

    Only the controller's OperationState determines update/gradient behavior.
    This loop fences a live private channel and verifies staged inputs. A lost
    execute reply cannot authorize a retry. The owner tears down an invalid or
    disconnected worker; restoration requires a new incarnation and checkpoint.
    Execute must return only after required terminal transfers, with an optional
    immutable host artifact reference; it must not return an execution-buffer view.
    """

    def __init__(
        self,
        scope: WorkerScope,
        bind: Callable[[WorkerCall], BoundWorkerCall],
        execute: Callable[
            [BoundWorkerCall], ArtifactRef | WorkerArtifacts | RankSaved | None
        ],
        budget: BufferBudget,
    ) -> None:
        self.scope, self._bind, self._execute = scope, bind, execute
        self._budget = budget
        self._next = 1
        self._last_sequence = 0
        self._pending: tuple[WorkerTicket, BoundWorkerCall] | None = None
        self.valid = True

    def handle(self, message: Record) -> WorkerReply:
        if not self.valid:
            raise RuntimeError("worker channel is invalid; never replay execution")
        try:
            if isinstance(message, WorkerCall):
                return self._stage(message)
            if isinstance(message, WorkerControl):
                return self._control(message)
            raise ProtocolError("worker expects a call or control record")
        except BaseException:
            self.invalidate()
            raise

    def _stage(self, command: WorkerCall) -> WorkerReply:
        if (
            command.scope != self.scope
            or command.invocation != self._next
            or command.request.identity.sequence <= self._last_sequence
            or self._pending is not None
        ):
            raise ProtocolError("stale, repeated or out-of-order worker invocation")
        identity = ticket(command)
        self._last_sequence = command.request.identity.sequence
        try:
            bound = self._bind(command)
            if bound.command != command:
                raise ProtocolError("worker binding changed the received command")
            bound.verify_inputs(self._budget)
            if (command.request.operation == "save") != (bound.checkpoint is not None):
                raise ValueError("worker has no prepared checkpoint encoder")
            if (command.request.operation == "restore") != (bound.restore is not None):
                raise ValueError("worker has no prepared checkpoint loader")
            ready = WorkerReply(
                identity, "ready", "not_started", checkpoint=bound.checkpoint
            )
            CODEC.encode(ready)
            if bound.checkpoint is not None:
                info = command.checkpoint
                assert info is not None
                saved = RankSaved("0" * 64, bound.checkpoint.confirmation(info))
                reference = bound.checkpoint.reference(
                    f"{checkpoint_id(info)}.rank.{bound.checkpoint.rank}", saved.sha256
                ).archive
                CODEC.encode(
                    WorkerReply(
                        identity,
                        "completed",
                        "completed",
                        result=reference,
                        saved=saved,
                    )
                )
        except Exception as error:
            # No execution token was exposed. The controller can abort other
            # ready participants and preserve live state after every reply.
            self._next += 1
            return WorkerReply(
                identity, "failed", "not_started", message=_message(error)
            )
        self._pending = identity, bound
        return ready

    def _control(self, control: WorkerControl) -> WorkerReply:
        if self._pending is None or self._pending[0] != control.ticket:
            raise ProtocolError("control does not match a ready worker invocation")
        identity, bound = self._pending
        # Consume before calling any mutating operation, even if it raises or
        # the transport fails while writing the eventual completion response.
        self._pending = None
        self._next += 1
        if control.action == "abort":
            return WorkerReply(identity, "aborted", "not_started")
        try:
            value = self._execute(bound)
            saved = None
            trace = trace_issue = None
            execution_seconds = None
            result: ArtifactRef | None
            if isinstance(value, RankSaved):
                saved = value
                description, info = bound.checkpoint, bound.command.checkpoint
                if description is None or info is None:
                    raise ValueError("worker returned an unprepared checkpoint")
                description.confirm(info, saved.manifest)
                result = description.reference(
                    f"{checkpoint_id(info)}.rank.{description.rank}", saved.sha256
                ).archive
            else:
                if bound.checkpoint is not None:
                    raise ValueError("worker omitted its checkpoint completion")
                if isinstance(value, WorkerArtifacts):
                    execution_seconds = value.execution_seconds
                    result, trace, trace_issue = (
                        value.result,
                        value.trace,
                        value.trace_issue,
                    )
                else:
                    result = value
            if bound.command.request.operation == "close":
                # serve writes this final receipt before leaving its normal
                # installed-worker cleanup scope. No further call can execute.
                self.invalidate()
            return WorkerReply(
                identity,
                "completed",
                "completed",
                result=result,
                saved=saved,
                trace=trace,
                trace_issue=trace_issue,
                execution_seconds=execution_seconds,
            )
        except WorkerOutputError as error:
            return WorkerReply(identity, "failed", "completed", message=_message(error))
        except Exception as error:
            self.invalidate()
            return WorkerReply(identity, "failed", "unknown", message=_message(error))

    def invalidate(self) -> None:
        self.valid = False
        self._pending = None

    def serve(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """Block without an idle timeout; the owning agent manages process lifetime."""
        try:
            while self.valid:
                message = read_message(reader)
                if message is None:
                    return
                write_message(writer, self.handle(message))
        finally:
            # EOF, truncation, broken replies and normal shutdown never leave a
            # reusable, unfenced mutable worker behind the lost connection.
            self.invalidate()
