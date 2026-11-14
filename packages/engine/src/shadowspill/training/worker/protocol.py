"""Private controller/worker metadata; tensor bytes and Python code stay separate."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from shadowspill_client._records import Record, integer, names, record, text
from shadowspill_client.protocol.codec import RecordCodec
from shadowspill_client.protocol.requests import (
    STATE_OPERATIONS,
    ArtifactRef,
    OperationRequest,
    identifier,
)


class WorkerOutputError(RuntimeError):
    """Execution and cleanup completed, but required host publication failed.

    Only the execution adapter may establish this boundary, after synchronizing
    terminal work and releasing borrowed device output views. It does not permit
    replay of the completed invocation.
    """


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
class WorkerStartup(Record):
    """CPU preflight evidence for one admitted bundle and assigned worker."""

    scope: WorkerScope
    bundle: ArtifactRef
    initialization_sha256: str
    logical_by_worker: Sequence[int]
    device_bytes: int
    spill_bytes: int
    staging_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(WorkerScope, self.scope, "scope"))
        object.__setattr__(self, "bundle", record(ArtifactRef, self.bundle, "bundle"))
        if (
            not isinstance(self.initialization_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.initialization_sha256) is None
        ):
            raise ValueError(
                "initialization digest must contain 64 lowercase hex digits"
            )
        ranks = self.logical_by_worker
        if (
            not isinstance(ranks, (tuple, list))
            or not 0 < len(ranks) <= 65536
            or any(type(rank) is not int for rank in ranks)
            or sorted(ranks) != list(range(len(ranks)))
            or self.scope.rank >= len(ranks)
        ):
            raise ValueError("startup requires a complete logical worker permutation")
        object.__setattr__(self, "logical_by_worker", tuple(ranks))
        for name in ("device_bytes", "spill_bytes", "staging_bytes"):
            integer(getattr(self, name), name, minimum=1)


@dataclass(frozen=True)
class WorkerBoot(Record):
    startup: WorkerStartup
    phase: Literal["prepared", "start", "ready"]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "startup", record(WorkerStartup, self.startup, "startup")
        )
        if self.phase not in ("prepared", "start", "ready"):
            raise ValueError("unknown worker startup phase")


@dataclass(frozen=True)
class WorkerLaunch(Record):
    """Trusted agent configuration on the inherited private channel only.

    Definition packages are installed deployment configuration, never requested
    Python imports. Initialization references are granted by the artifact owner.
    Filesystem roots remain local command arguments supplied by the node agent.
    """

    startup: WorkerStartup
    packages: Sequence[str]
    weights: Sequence[ArtifactRef] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "startup", record(WorkerStartup, self.startup, "startup")
        )
        packages = names(self.packages, "installed definition packages")
        if len(packages) > 64 or any(len(name) > 128 for name in packages):
            raise ValueError("worker definition packages exceed their bound")
        object.__setattr__(self, "packages", packages)
        if not isinstance(self.weights, (tuple, list)) or len(self.weights) > 1024:
            raise ValueError("worker weights require bounded artifact references")
        weights = tuple(record(ArtifactRef, ref, "weights") for ref in self.weights)
        if len({self.startup.bundle.id, *(ref.id for ref in weights)}) != 1 + len(
            weights
        ):
            raise ValueError("worker initialization artifact IDs must be distinct")
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class WorkerCall(Record):
    """A saved candidate and the sole coordinator's queue-head state decision.

    Invocation order is independent of submission sequence: rejected requests
    can leave gaps in the latter. Frames follow the saved logical-rank order.
    No path, arbitrary Plan selector, model import or state reset is accepted.
    """

    scope: WorkerScope
    invocation: int
    request: OperationRequest
    candidate: str | None
    skipped: bool
    accumulate_gradients: bool
    frames: Sequence[ArtifactRef] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(WorkerScope, self.scope, "scope"))
        object.__setattr__(
            self, "request", record(OperationRequest, self.request, "request")
        )
        integer(self.invocation, "worker invocation", minimum=1)
        if self.request.operation in STATE_OPERATIONS:
            if self.candidate is not None or self.skipped or self.accumulate_gradients:
                raise ValueError("state commands have no compute candidate or flags")
        else:
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
        if (
            self.request.operation == "optim_step"
            or self.request.operation in STATE_OPERATIONS
        ) and frames:
            raise ValueError("commands without batches have no input frames")
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
        "worker.launch": WorkerLaunch,
        "worker.boot": WorkerBoot,
        "worker.call": WorkerCall,
        "worker.control": WorkerControl,
        "worker.reply": WorkerReply,
    }
)
