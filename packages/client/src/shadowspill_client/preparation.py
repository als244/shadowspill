"""Measurement and planning jobs, separate from live training clients."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ._async import CallFuture
from ._progress import ProgressDisplay
from ._records import Value, mapping
from .artifacts import ArtifactHandle
from .creation import CreationError, _info
from .plans import PlansHandle
from .protocol.creation import TrainingCreation
from .protocol.preview import TrainingPreview
from .protocol.requests import identifier
from .protocol.runs import RunInfo, RunKind, organization
from .protocol.setup import TrainingSetup
from .protocol.training import ResolvedTraining

if TYPE_CHECKING:
    from .service import ServiceClient


@dataclass(frozen=True)
class PreparationHandle:
    """Completed measured cases; this handle owns no mutable training state."""

    info: RunInfo
    _service: ServiceClient

    @property
    def reuse(self) -> Mapping[str, Value]:
        """Portable provenance for ``Preparation(reuse=prepared.reuse)``."""
        return mapping(
            {
                "measurements": {
                    "session_id": self.info.session_id,
                    "run_id": self.info.run_id,
                },
            },
            "preparation reuse",
        )

    def export(self) -> CallFuture[ArtifactHandle]:
        """Obtain the immutable ZIP containing all cases and their manifest."""
        reference = self.info.preparation_export
        if self.info.state != "prepared" or reference is None:
            raise ValueError("preparation has no completed measurement export")
        return self._service._work.submit(
            lambda: ArtifactHandle(reference, self._service)
        )


class _JobFuture[T](CallFuture[T]):
    def __init__(self, session_id: str, run_id: str) -> None:
        super().__init__()
        self.session_id, self.run_id = session_id, run_id


class PreparationFuture(_JobFuture[PreparationHandle]):
    """A measurement job whose result contains an export for offline planning."""


class PlanningFuture(_JobFuture[PlansHandle]):
    """A planning job whose result contains Plans and their inspection report."""


def _completed(
    service: ServiceClient, run_id: str, kind: RunKind, display: ProgressDisplay
) -> RunInfo:
    route = f"/v1/runs/{service.session_id}/{run_id}"
    while True:
        service._work.check()
        reply = service._transport.request("GET", route)
        info = _info(service, reply.value, run_id)
        if info.kind != kind:
            raise ValueError(f"run is not a {kind} job")
        display.update(
            service,
            info,
            final=info.state in ("failed", "stopped")
            or (reply.status == 200 and info.state != "preparing"),
        )
        if info.state in ("failed", "stopped"):
            raise CreationError(info)
        if reply.status == 200 and info.state in ("prepared", "planned"):
            return info
        service._work.pause()


def _submit[T](
    service: ServiceClient,
    *,
    future: _JobFuture[T],
    result: Callable[[RunInfo], T],
    kind: RunKind,
    config: ResolvedTraining | None,
    wait_for_resources: bool,
    name: str | None,
    labels: Mapping[str, str] | None,
    progress: bool,
    setup: dict[str, Any],
) -> _JobFuture[T]:
    display = ProgressDisplay(progress)
    settings = service._setup(config, setup)
    frozen_labels = organization(name, {} if labels is None else labels)
    if type(wait_for_resources) is not bool:
        raise TypeError("wait_for_resources must be boolean")

    def create() -> T:
        resolved = (
            service._request(
                "POST", "/v1/training/describe", TrainingPreview, settings
            ).config
            if isinstance(settings, TrainingSetup)
            else settings
        )
        creation = TrainingCreation(
            resolved,
            wait_for_resources=wait_for_resources,
            name=name,
            labels=frozen_labels,
            kind=kind,
        )
        _info(
            service,
            service._transport.request(
                "POST", f"/v1/runs/{service.session_id}/{future.run_id}", creation
            ).value,
            future.run_id,
        )
        return result(_completed(service, future.run_id, kind, display))

    service._work.submit(create, future=future)
    return future


def prepare_training(
    service: ServiceClient,
    *,
    config: ResolvedTraining | None = None,
    wait_for_resources: bool = False,
    name: str | None = None,
    labels: Mapping[str, str] | None = None,
    progress: bool = False,
    **setup: Any,
) -> PreparationFuture:
    future = PreparationFuture(service.session_id, uuid4().hex)
    _submit(
        service,
        future=future,
        result=lambda info: PreparationHandle(info, service),
        kind="preparation",
        config=config,
        wait_for_resources=wait_for_resources,
        name=name,
        labels=labels,
        progress=progress,
        setup=setup,
    )
    return future


def _plans(service: ServiceClient, info: RunInfo) -> PlansHandle:
    assert info.plans is not None
    return PlansHandle(info.plans, service, info.run_id)


def plan_training(
    service: ServiceClient,
    *,
    config: ResolvedTraining | None = None,
    wait_for_resources: bool = False,
    name: str | None = None,
    labels: Mapping[str, str] | None = None,
    progress: bool = False,
    **setup: Any,
) -> PlanningFuture:
    """Build/profile/plan without loading training state or persistent workers."""
    future = PlanningFuture(service.session_id, uuid4().hex)
    _submit(
        service,
        future=future,
        result=lambda info: _plans(service, info),
        kind="planning",
        config=config,
        wait_for_resources=wait_for_resources,
        name=name,
        labels=labels,
        progress=progress,
        setup=setup,
    )
    return future


def get_preparation(
    service: ServiceClient, run_id: str, *, progress: bool = False
) -> PreparationFuture:
    """Reattach read-only after disconnect, without recreating or replaying a job."""
    identifier(run_id, "preparation run ID")
    display = ProgressDisplay(progress)
    future = PreparationFuture(service.session_id, run_id)
    service._work.submit(
        lambda: PreparationHandle(
            _completed(service, run_id, "preparation", display), service
        ),
        future=future,
    )
    return future


def get_planning(
    service: ServiceClient, run_id: str, *, progress: bool = False
) -> PlanningFuture:
    """Rejoin an accepted planning job without rerunning its phases."""
    identifier(run_id, "planning run ID")
    display = ProgressDisplay(progress)
    future = PlanningFuture(service.session_id, run_id)
    service._work.submit(
        lambda: _plans(service, _completed(service, run_id, "planning", display)),
        future=future,
    )
    return future
