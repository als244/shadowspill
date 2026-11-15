"""Controller endpoint for one private worker stream; no resend or reconnect."""

from typing import BinaryIO, Literal, NoReturn

from shadowspill_client.protocol.codec import ProtocolError

from .binary import BinarySink, receive_stream
from .channel import read_message, write_message
from .protocol import (
    WorkerCall,
    WorkerControl,
    WorkerReply,
    WorkerScope,
    WorkerTicket,
    ticket,
)


class WorkerConnection:
    """A single controller owns both streams and serializes their use.

    Sending and receiving execution are separate so the controller can release
    all collective participants before waiting for any of them. An uncertain
    send/reply invalidates this endpoint and is never retried on this channel.
    """

    def __init__(self, scope: WorkerScope, reader: BinaryIO, writer: BinaryIO) -> None:
        self.scope, self._reader, self._writer = scope, reader, writer
        self.next_invocation = 1
        self.valid = True
        self._pending: WorkerTicket | None = None
        self._phase: Literal["staging", "ready", "executing", "aborting"] | None = None
        self._stream_size: int | None = None

    def check(self, command: WorkerCall) -> None:
        if not self.valid:
            raise RuntimeError("worker connection is invalid; never resend execution")
        if (
            command.scope != self.scope
            or command.invocation != self.next_invocation
            or self._pending is not None
        ):
            raise ProtocolError("command does not match the idle worker connection")
        ticket(command)  # Enforce the complete bounded envelope before any send.

    def stage(self, command: WorkerCall) -> None:
        self.check(command)
        self._pending, self._phase = ticket(command), "staging"
        try:
            write_message(self._writer, command)
        except BaseException:
            self.valid = False
            raise

    def staged(self) -> WorkerReply:
        self._require("staging")
        reply = self._receive()
        if reply.status not in ("ready", "failed") or reply.effect != "not_started":
            self._invalid("worker staging reply claims unexpected execution effects")
        if reply.status == "ready":
            self._phase = "ready"
        else:
            self._consumed()
        return reply

    def release(self, action: Literal["execute", "abort"]) -> None:
        self._require("ready")
        assert self._pending is not None
        control = WorkerControl(self._pending, action)
        self._phase = "executing" if action == "execute" else "aborting"
        try:
            write_message(self._writer, control)
        except BaseException:
            self.valid = False
            raise

    def finished(self) -> WorkerReply:
        if self._phase not in ("executing", "aborting"):
            raise RuntimeError("worker has no released invocation to finish")
        phase = self._phase
        reply = self._receive()
        expected = (
            {
                ("completed", "completed"),
                ("failed", "completed"),
                ("failed", "unknown"),
            }
            if phase == "executing"
            else {("aborted", "not_started")}
        )
        if (reply.status, reply.effect) not in expected:
            self._invalid("worker completion reply differs from the released action")
        if (
            self._stream_size is not None
            and reply.status == "completed"
            and (reply.result is None or reply.result.size_bytes != self._stream_size)
        ):
            self._invalid("worker result differs from its completed binary stream")
        self._consumed()
        if reply.effect == "unknown":
            self.valid = False
        return reply

    def receive_stream(self, destination: BinarySink, *, size_bytes: int) -> None:
        """Receive one admitted output, then require its ordinary completion reply.

        Call after release, with persistent storage already reserved. Reading or
        writing failure invalidates this channel; the group must confirm shutdown
        before disposing its failed producer. No reconnect or replay is attempted.
        """
        self._require("executing")
        if self._stream_size is not None:
            raise RuntimeError("worker invocation already streamed its output")
        try:
            receive_stream(self._reader, destination, size_bytes=size_bytes)
            self._stream_size = size_bytes
        except BaseException:
            self.valid = False
            raise

    def _require(self, phase: str) -> None:
        if not self.valid or self._phase != phase:
            raise RuntimeError(f"worker is not {phase}; never resend execution")

    def _receive(self) -> WorkerReply:
        if not self.valid:
            raise RuntimeError("worker connection is invalid; never resend execution")
        try:
            reply = read_message(self._reader)
            if not isinstance(reply, WorkerReply) or reply.ticket != self._pending:
                self._invalid("worker reply is absent or belongs to another invocation")
            return reply
        except BaseException:
            self.valid = False
            raise

    def _consumed(self) -> None:
        self._pending = self._phase = None
        self._stream_size = None
        self.next_invocation += 1

    def _invalid(self, message: str) -> NoReturn:
        self.valid = False
        raise ProtocolError(message)

    def close(self) -> None:
        self.valid = False
        try:
            self._writer.close()
        finally:
            self._reader.close()
