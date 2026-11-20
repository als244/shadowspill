"""Private controller/worker metadata; tensor bytes and Python code stay separate."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from shadowspill_client._records import Record, integer, names, number, record, text
from shadowspill_client.protocol.checkpoints import SaveInfo
from shadowspill_client.protocol.codec import RecordCodec
from shadowspill_client.protocol.requests import (
    STATE_OPERATIONS,
    ArtifactRef,
    OperationRequest,
    identifier,
)
from shadowspill_client.protocol.results import ArtifactIssue

from .checkpoint_records import RankRestore, RankSave, RankSaved, RankWeights
from .conversion_records import (
    ConversionCommand,
    ConversionFailed,
    ConversionPrepared,
    ConversionRequest,
    ConversionWritten,
)


class WorkerOutputError(RuntimeError):
    """Execution and cleanup completed, but required host publication failed.

    Only the execution adapter may establish this boundary, after synchronizing
    terminal work and releasing borrowed device output views. It does not permit
    replay of the completed invocation.
    """


@dataclass(frozen=True)
class WorkerScope(Record):
    """Identity of one training worker process within a run.

    The service makes one for each worker when it starts a run.

    Attributes
    ----------
    run_id : str
        The run's ID.
    server_incarnation : str
        `Server.incarnation` of the service that started the worker.
    worker_incarnation : str
        ID of this worker process, new at each start.
    rank : int
        The worker's logical rank.
    """

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
    """What one worker needs, resolved before the service admits it.

    The service builds one for each worker when a run starts, from the run's
    Plans and saved state, before it claims any device; `WorkerHost.peak_bytes`
    reads it.

    Attributes
    ----------
    scope : WorkerScope
        The worker's identity.
    bundle : ArtifactRef
        The run's Plan bundle.
    initialization_sha256 : str
        SHA-256, in 64 lowercase hex digits, of what the worker starts from:
        checkpoint, weights, initializer, frozen base and staging size.
    logical_by_worker : tuple of int
        The logical rank of each worker process, by position: a permutation
        of the run's ranks.
    device_bytes : int
        Execution pool of the worker's Plan on its device, in bytes.
    spill_bytes : int
        Capacity of the worker's spill pool, in bytes.
    staging_bytes : int
        Host staging, in bytes, for the worker's inputs, outputs and
        initialization.
    diagnostics_bytes : int
        Host memory, in bytes, of the largest runtime trace among its Plans.
    """

    scope: WorkerScope
    bundle: ArtifactRef
    initialization_sha256: str
    logical_by_worker: Sequence[int]
    device_bytes: int
    spill_bytes: int
    staging_bytes: int
    diagnostics_bytes: int = 0

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
        integer(self.diagnostics_bytes, "worker diagnostics bytes")


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
    """Trusted node daemon configuration on the inherited private channel only.

    Component packages are installed deployment configuration, never requested
    Python imports. Initialization references are granted by the artifact owner.
    Filesystem roots remain local command arguments supplied by the node daemon.
    """

    startup: WorkerStartup
    packages: Sequence[str]
    weights: Sequence[ArtifactRef] = ()
    restore: RankRestore | None = None
    initial_weights: RankWeights | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "startup", record(WorkerStartup, self.startup, "startup")
        )
        packages = names(self.packages, "installed component packages")
        if len(packages) > 64 or any(len(name) > 128 for name in packages):
            raise ValueError("worker component packages exceed their bound")
        object.__setattr__(self, "packages", packages)
        if not isinstance(self.weights, (tuple, list)) or len(self.weights) > 1024:
            raise ValueError("worker weights require bounded artifact references")
        weights = tuple(record(ArtifactRef, ref, "weights") for ref in self.weights)
        if len({self.startup.bundle.id, *(ref.id for ref in weights)}) != 1 + len(
            weights
        ):
            raise ValueError("worker initialization artifact IDs must be distinct")
        object.__setattr__(self, "weights", weights)
        if self.restore is not None and self.initial_weights is not None:
            raise ValueError("worker startup cannot combine weights and a checkpoint")
        saved_types: tuple[tuple[str, type[RankRestore] | type[RankWeights]], ...] = (
            ("restore", RankRestore),
            ("initial_weights", RankWeights),
        )
        for name, kind in saved_types:
            saved = getattr(self, name)
            if saved is None:
                continue
            saved = record(kind, saved, name)
            object.__setattr__(self, name, saved)
            base = saved.info.frozen_base
            required = () if base is None or base.artifact is None else (base.artifact,)
            if (
                weights != required
                or saved.rank.rank != self.startup.scope.rank
                or saved.info.world_size != len(self.startup.logical_by_worker)
            ):
                raise ValueError(
                    "saved-state startup requires its rank and exact frozen base; "
                    "no fresh weights"
                )


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
    accumulate_gradients: bool
    frames: Sequence[ArtifactRef] = ()
    checkpoint: SaveInfo | None = None
    restore: RankRestore | None = None
    history_id: str | None = None
    completed_updates: int = 0
    #: Tokens the request supplies and tokens its candidate schedules, over
    #: the whole batch; both absent when the request has no token count.
    batch_tokens: int | None = None
    scheduled_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", record(WorkerScope, self.scope, "scope"))
        if (self.batch_tokens is None) != (self.scheduled_tokens is None):
            raise ValueError("supplied and scheduled tokens come together")
        if self.batch_tokens is not None:
            assert self.scheduled_tokens is not None
            integer(self.batch_tokens, "request tokens")
            integer(self.scheduled_tokens, "scheduled tokens")
            if self.batch_tokens > self.scheduled_tokens:
                raise ValueError("a request cannot supply more tokens than scheduled")
        object.__setattr__(
            self, "request", record(OperationRequest, self.request, "request")
        )
        integer(self.invocation, "worker invocation", minimum=1)
        integer(self.completed_updates, "completed updates before invocation")
        if self.history_id is not None:
            identifier(self.history_id, "history ID")
        if self.request.runtime_trace and self.history_id is None:
            raise ValueError("traced calls require their coordinator history")
        if self.restore is not None:
            object.__setattr__(
                self, "restore", record(RankRestore, self.restore, "restore")
            )
            if self.restore.rank.rank != self.scope.rank:
                raise ValueError("restore archive belongs to another logical rank")
        if (self.request.operation == "restore") != (self.restore is not None):
            raise ValueError("restore requires its accepted immutable rank archive")
        if self.checkpoint is not None:
            object.__setattr__(
                self,
                "checkpoint",
                record(SaveInfo, self.checkpoint, "checkpoint"),
            )
        if (self.request.operation == "save") != (self.checkpoint is not None):
            raise ValueError(
                "checkpoint save requires the coordinator's quiescent boundary"
            )
        if self.checkpoint is not None:
            info, settings = self.checkpoint, self.request.save
            assert settings is not None
            if (
                info.run_id != self.scope.run_id
                or info.request_id != self.request.identity.request_id
                or self.scope.rank >= info.world_size
                or info.weights != settings.weights
                or info.metadata != settings.metadata
                or info.kind != settings.kind
                or info.format != settings.format
            ):
                raise ValueError("checkpoint boundary differs from its worker request")
        if self.request.operation in STATE_OPERATIONS:
            if self.candidate is not None or self.accumulate_gradients:
                raise ValueError("state commands have no compute candidate or flags")
        else:
            identifier(self.candidate, "prepared candidate")
        identity = self.request.identity
        if (
            identity.run_id != self.scope.run_id
            or identity.server_incarnation != self.scope.server_incarnation
        ):
            raise ValueError("worker scope differs from its request owner")
        if type(self.accumulate_gradients) is not bool:
            raise TypeError("worker state decisions must be boolean")
        if self.accumulate_gradients and self.request.operation != "forward_backward":
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
    checkpoint: RankSave | None = None
    saved: RankSaved | None = None
    trace: ArtifactRef | None = None
    trace_issue: ArtifactIssue | None = None
    execution_seconds: float | None = None

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
        if self.execution_seconds is not None:
            number(self.execution_seconds, "worker execution seconds", minimum=0)
            if self.effect != "completed":
                raise ValueError("execution timing requires confirmed effects")
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
        if self.checkpoint is not None:
            object.__setattr__(
                self, "checkpoint", record(RankSave, self.checkpoint, "checkpoint")
            )
            if self.status != "ready" or self.checkpoint.rank != self.ticket.scope.rank:
                raise ValueError("only a ready worker describes its checkpoint rank")
        if self.saved is not None:
            object.__setattr__(self, "saved", record(RankSaved, self.saved, "saved"))
            if (
                self.status != "completed"
                or self.result is None
                or self.result.sha256 != self.saved.sha256
            ):
                raise ValueError("saved checkpoint requires its completed artifact")
        if self.trace is not None:
            object.__setattr__(self, "trace", record(ArtifactRef, self.trace, "trace"))
        if self.trace_issue is not None:
            issue = record(ArtifactIssue, self.trace_issue, "trace issue")
            object.__setattr__(self, "trace_issue", issue)
            if len(issue.code) > 64 or len(issue.message) > 1024:
                raise ValueError("worker trace issue exceeds its bound")
        if self.trace is not None or self.trace_issue is not None:
            if self.status != "completed" or self.saved is not None:
                raise ValueError(
                    "only completed compute returns optional trace evidence"
                )
            if self.trace is not None and self.trace_issue is not None:
                raise ValueError("trace evidence cannot be both ready and failed")


@dataclass(frozen=True)
class WorkerArtifacts:
    """Completed device work, required output and optional trace publication."""

    result: ArtifactRef | None
    trace: ArtifactRef | None = None
    trace_issue: ArtifactIssue | None = None
    execution_seconds: float | None = None


CODEC = RecordCodec(
    {
        "worker.launch": WorkerLaunch,
        "worker.boot": WorkerBoot,
        "worker.call": WorkerCall,
        "worker.control": WorkerControl,
        "worker.reply": WorkerReply,
        "conversion.request": ConversionRequest,
        "conversion.prepared": ConversionPrepared,
        "conversion.command": ConversionCommand,
        "conversion.written": ConversionWritten,
        "conversion.failed": ConversionFailed,
    }
)
