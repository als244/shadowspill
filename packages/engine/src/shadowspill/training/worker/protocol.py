"""Private controller/worker metadata; tensor bytes and Python code stay separate."""

from collections.abc import Sequence
from dataclasses import dataclass

from shadowspill_client._records import Record, integer, record
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


CODEC = RecordCodec({"worker.call": WorkerCall})
