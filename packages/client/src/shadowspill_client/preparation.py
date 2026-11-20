"""Measurement and planning jobs, separate from live training clients."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ._async import CallFuture
from ._progress import ProgressDisplay
from ._records import Value, mapping, record
from .artifacts import ArtifactHandle
from .config import Resources
from .creation import CreationError, _automatic, _info
from .plans import PlansHandle
from .protocol.creation import TrainingCreation
from .protocol.preview import TrainingDescription
from .protocol.requests import identifier
from .protocol.runs import RunInfo, RunKind, organization
from .protocol.setup import TrainingSetup
from .protocol.training import ResolvedTraining

if TYPE_CHECKING:
    from .service import ServiceClient


@dataclass(frozen=True)
class PreparationHandle:
    """A completed measurement job and its measured cases.

    The value of the `PreparationFuture` that `ServiceClient.prepare_training`
    and `ServiceClient.get_preparation` return.

    Attributes
    ----------
    info : RunInfo
        The job's run record, in the ``"prepared"`` state.
    """

    info: RunInfo
    _service: ServiceClient

    @property
    def reuse(self) -> Mapping[str, Value]:
        """Reference to these measurements, for ``Preparation(reuse=...)``.

        A later preparation or training run given it reuses the compatible
        measured cases instead of measuring them again.
        """
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
        """Get the stored ZIP archive of the measured cases, for offline planning.

        The archive holds each case's phase archives and a ``manifest.json``
        naming every file, its checksum and the resolved settings.

        Returns
        -------
        CallFuture of ArtifactHandle
            The archive, downloaded with `ArtifactHandle.download`.

        Raises
        ------
        ValueError
            If the job has no completed export.
        """
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
    """`CallFuture` of a `PreparationHandle` for a measurement job.

    `ServiceClient.prepare_training` and `ServiceClient.get_preparation`
    return it. ``result()`` blocks until measurement finishes and returns the
    handle. A wait that times out raises `TimeoutError` and leaves the job
    running.

    Attributes
    ----------
    session_id : str
        The session the job belongs to.
    run_id : str
        The job's run ID, known at once. Pass it to
        `ServiceClient.get_preparation` to rejoin the job.
    """


class PlanningFuture(_JobFuture[PlansHandle]):
    """`CallFuture` of a `PlansHandle` for a planning job.

    `ServiceClient.plan_training` and `ServiceClient.get_planning` return it.
    ``result()`` blocks until planning finishes and returns the Plans. A wait
    that times out raises `TimeoutError` and leaves the job running.

    Attributes
    ----------
    session_id : str
        The session the job belongs to.
    run_id : str
        The job's run ID, known at once. Pass it to
        `ServiceClient.get_planning` to rejoin the job.
    """


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
    resources: Resources | None,
    setup: dict[str, Any],
) -> _JobFuture[T]:
    display = ProgressDisplay(progress)
    settings = service._setup(config, setup)
    frozen_labels = organization(name, {} if labels is None else labels)
    resource_request = (
        None if resources is None else record(Resources, resources, "resources")
    )
    if type(wait_for_resources) is not bool:
        raise TypeError("wait_for_resources must be boolean")

    def create() -> T:
        selected = resource_request
        if isinstance(settings, TrainingSetup) or _automatic(settings.memory):
            # Describing sizes automatic memory budgets on the selected workers.
            preview = service._describe(TrainingDescription.request(settings, selected))
            resolved, selected = preview.config, preview.resources
        else:
            resolved = settings
        creation = TrainingCreation(
            resolved,
            wait_for_resources=wait_for_resources,
            name=name,
            labels=frozen_labels,
            kind=kind,
            resources=selected,
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
    resources: Resources | None = None,
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
        resources=resources,
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
    resources: Resources | None = None,
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
        resources=resources,
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
