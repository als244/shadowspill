"""Private controller/worker metadata; tensor bytes and Python code stay separate."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from shadowspill_client._records import Record, integer, record, text
from shadowspill_client.protocol.codec import RecordCodec
from shadowspill_client.protocol.requests import ArtifactRef, ComputeRequest, identifier


@dataclass(frozen=True)
class WorkerScope(Record):
    run_id: str
    server_incarnation: str
    worker_incarnation: str
    rank: int

    def __post_init__(self) -> None:
        for name in ("run_id", "server_incarnation", "worker_incarnation"):
            identifier(getattr(self, name), name)
        integer(self.rank, "logical worker rank")


@dataclass(frozen=True)
class WorkerCall(Record):
    """A saved candidate and the sole coordinator's queue-head state decision.

    Invocation order is independent of submission sequence: rejected requests
    can leave gaps in the latter. Frames follow the saved logical-rank order.
    No path, arbitrary Plan selector, model import or state reset is accepted.
    """

    scope: WorkerScope
    invocation: int
    request: ComputeRequest
    candidate: str
    skipped: bool
    accumulate_gradients: bool
    frames: Sequence[ArtifactRef] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(WorkerScope, self.scope, "scope"))
        object.__setattr__(
            self, "request", record(ComputeRequest, self.request, "request")
        )
        integer(self.invocation, "worker invocation", minimum=1)
        identifier(self.candidate, "prepared candidate")
        identity = self.request.identity
        if (
            identity.run_id != self.scope.run_id
            or identity.server_incarnation != self.scope.server_incarnation
        ):
            raise ValueError("worker scope differs from its request owner")
        if (
            type(self.skipped) is not bool
            or type(self.accumulate_gradients) is not bool
        ):
            raise TypeError("worker state decisions must be boolean")
        if self.request.operation == "forward" and self.skipped:
            raise ValueError("forward cannot select a masked training form")
        if self.accumulate_gradients and (
            self.request.operation != "forward_backward" or self.skipped
        ):
            raise ValueError("only participating backward can accumulate gradients")
        if not isinstance(self.frames, (tuple, list)) or len(self.frames) > 65536:
            raise ValueError("worker frames must be a bounded list")
        frames = tuple(record(ArtifactRef, item, "frame") for item in self.frames)
        if len({item.id for item in frames}) != len(frames):
            raise ValueError("worker frame IDs must be distinct")
        if self.request.operation == "optim_step" and frames:
            raise ValueError("optimizer-only commands have no input frames")
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class WorkerTicket(Record):
    scope: WorkerScope
    invocation: int
    request_id: str
    call_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(WorkerScope, self.scope, "scope"))
        integer(self.invocation, "worker invocation", minimum=1)
        identifier(self.request_id, "request ID")
        if (
            not isinstance(self.call_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.call_sha256) is None
        ):
            raise ValueError("worker call digest must contain 64 lowercase hex digits")


def ticket(command: WorkerCall) -> WorkerTicket:
    return WorkerTicket(
        command.scope,
        command.invocation,
        command.request.identity.request_id,
        hashlib.sha256(CODEC.encode(command)).hexdigest(),
    )


@dataclass(frozen=True)
class WorkerControl(Record):
    ticket: WorkerTicket
    action: Literal["execute", "abort"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticket", record(WorkerTicket, self.ticket, "ticket"))
        if self.action not in ("execute", "abort"):
            raise ValueError("unknown worker control action")


@dataclass(frozen=True)
class WorkerReply(Record):
    ticket: WorkerTicket
    status: Literal["ready", "aborted", "completed", "failed"]
    effect: Literal["not_started", "completed", "unknown"]
    result: ArtifactRef | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticket", record(WorkerTicket, self.ticket, "ticket"))
        if self.status not in ("ready", "aborted", "completed", "failed"):
            raise ValueError("unknown worker reply status")
        if self.effect not in ("not_started", "completed", "unknown"):
            raise ValueError("unknown worker execution effect")
        if self.status in ("ready", "aborted") and self.effect != "not_started":
            raise ValueError("staging cannot claim execution effects")
        if self.status == "completed" and self.effect != "completed":
            raise ValueError("completed worker replies require confirmed effects")
        if self.message is not None:
            text(self.message, "worker failure message")
            if len(self.message) > 1024:
                raise ValueError("worker failure message exceeds 1024 characters")
        if (self.status == "failed") != (self.message is not None):
            raise ValueError("only failed worker replies have a failure message")
        if self.result is not None:
            object.__setattr__(
                self, "result", record(ArtifactRef, self.result, "result")
            )
            if self.status != "completed":
                raise ValueError("only completed workers return output artifacts")


CODEC = RecordCodec(
    {
        "worker.call": WorkerCall,
        "worker.control": WorkerControl,
        "worker.reply": WorkerReply,
    }
)
