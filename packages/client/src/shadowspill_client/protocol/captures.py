"""Portable per-rank Nsight evidence, independent of live workers and CUDA."""

from dataclasses import dataclass
from typing import Literal

from .._records import Record, integer, record, text
from .requests import ArtifactRef, identifier

NSIGHT_MEDIA_TYPE = "application/x-nsys-rep"
WINDOW_BYTES = 16 << 10


def rank_name(rank: int) -> str:
    integer(rank, "capture rank")
    if rank >= 65536:
        raise ValueError("capture rank exceeds the worker bound")
    return f"rank-{rank}"


@dataclass(frozen=True)
class NsightWindow(Record):
    """The updates an Nsight Systems capture covers, and how far it got.

    The capture starts with update ``start_step``'s first ``"step"`` or
    ``"forward_backward"`` request and covers ``steps`` consecutive updates
    of one training history. It never restarts once stopped.
    `NsightManifest.window` holds it.

    Parameters
    ----------
    start_step : int
        One-based optimizer update the capture starts at.
    steps : int
        Number of consecutive updates to capture, at least one.
    state : str, default "pending"
        ``"pending"`` (not started), ``"running"`` (started, not yet
        complete), ``"complete"`` (every selected update captured),
        ``"incomplete"`` (stopped early, as when the training history changed
        or the worker failed) or ``"missing"`` (never started, as when update
        ``start_step`` ran without the capture or the worker reported no
        progress).
    completed_updates : int, default 0
        Updates captured, at most ``steps``.
    history_id : str, optional
        Training history of the captured updates; None before the capture
        starts.
    first_request : str, optional
        ID of the request that started the capture; None before it starts.
    last_request : str, optional
        ID of the request that completed the last captured update; None
        before one completes.
    error : str, optional
        Why the window is incomplete or missing, at most 1024 characters; set
        exactly when ``state`` is ``"incomplete"`` or ``"missing"``.
    """

    start_step: int
    steps: int
    state: Literal["pending", "running", "complete", "incomplete", "missing"] = (
        "pending"
    )
    completed_updates: int = 0
    history_id: str | None = None
    first_request: str | None = None
    last_request: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        integer(self.start_step, "capture start step", minimum=1)
        integer(self.steps, "capture steps", minimum=1)
        integer(self.completed_updates, "captured updates")
        if self.state not in (
            "pending",
            "running",
            "complete",
            "incomplete",
            "missing",
        ):
            raise ValueError("unknown Nsight window state")
        for name in ("history_id", "first_request", "last_request"):
            if getattr(self, name) is not None:
                identifier(getattr(self, name), name)
        if self.completed_updates > self.steps:
            raise ValueError("capture completed more updates than selected")
        if self.state in ("pending", "missing") and (
            self.completed_updates
            or self.first_request is not None
            or self.last_request is not None
        ):
            raise ValueError("unstarted capture cannot contain update evidence")
        if self.state in ("running", "complete") and self.first_request is None:
            raise ValueError("started capture requires its first request")
        if self.completed_updates and self.last_request is None:
            raise ValueError("captured updates require their last request")
        if self.state == "complete" and self.completed_updates != self.steps:
            raise ValueError("complete capture requires every selected update")
        if self.error is not None and len(text(self.error, "capture error")) > 1024:
            raise ValueError("capture error exceeds its bound")
        if (self.state in ("incomplete", "missing")) != (self.error is not None):
            raise ValueError("incomplete/missing windows require an explanation")


@dataclass(frozen=True)
class NsightManifest(Record):
    """One worker's Nsight Systems capture record.

    `NsightCapture.record` holds it. The service records it as
    ``"pending"`` when the workers start and finalizes it after the worker
    exits; until then ``window`` shows only the selected updates.

    Parameters
    ----------
    session_id : str
        Session that owns the run.
    run_id : str
        The run.
    rank : int
        The worker's rank, below 65536.
    plans : ArtifactRef
        The run's execution Plans.
    window : NsightWindow
        The selected updates and how far the capture got.
    status : str, default "pending"
        ``"pending"`` (not finalized), ``"ready"`` (``report`` holds the
        complete window), ``"failed"`` (the collector failed, the window was
        not completed, or the report could not be stored) or ``"missing"``
        (no report was produced, or the run ended without finalizing the
        capture).
    report : ArtifactRef, optional
        The native report file (``application/x-nsys-rep``), nonempty. Set
        when ``status`` is ``"ready"`` and possibly when it is ``"failed"``;
        None otherwise.
    error : str, optional
        Why the capture failed or is missing, at most 1024 characters; set
        exactly when ``status`` is ``"failed"`` or ``"missing"``.
    """

    session_id: str
    run_id: str
    rank: int
    plans: ArtifactRef
    window: NsightWindow
    status: Literal["pending", "ready", "failed", "missing"] = "pending"
    report: ArtifactRef | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        identifier(self.session_id, "capture session ID")
        identifier(self.run_id, "capture run ID")
        rank_name(self.rank)
        object.__setattr__(
            self, "plans", record(ArtifactRef, self.plans, "capture Plans")
        )
        object.__setattr__(
            self, "window", record(NsightWindow, self.window, "capture window")
        )
        if self.status not in ("pending", "ready", "failed", "missing"):
            raise ValueError("unknown Nsight report status")
        if self.report is not None:
            object.__setattr__(
                self, "report", record(ArtifactRef, self.report, "Nsight report")
            )
            if (
                self.report.media_type != NSIGHT_MEDIA_TYPE
                or self.report.size_bytes == 0
            ):
                raise ValueError("Nsight report requires nonempty native report bytes")
        if self.status == "ready" and (
            self.window.state != "complete"
            or self.report is None
            or self.window.history_id is None
        ):
            raise ValueError("ready report requires a complete captured window")
        if self.status in ("pending", "missing") and self.report is not None:
            raise ValueError("pending/missing report cannot contain ready bytes")
        if self.error is not None and len(text(self.error, "Nsight error")) > 1024:
            raise ValueError("Nsight error exceeds its bound")
        if (self.status in ("failed", "missing")) != (self.error is not None):
            raise ValueError("failed/missing reports require a bounded explanation")
