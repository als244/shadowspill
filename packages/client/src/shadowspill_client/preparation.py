"""Measurement jobs and exports, separate from live training clients."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ._async import CallFuture
from ._progress import ProgressDisplay
from ._records import Value, mapping
from .artifacts import ArtifactHandle
from .creation import CreationError, _info
from .protocol.creation import TrainingCreation
from .protocol.preview import TrainingPreview
from .protocol.requests import identifier
from .protocol.runs import RunInfo, organization
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


class PreparationFuture(CallFuture[PreparationHandle]):
    def __init__(self, session_id: str, run_id: str) -> None:
        super().__init__()
        self.session_id, self.run_id = session_id, run_id


def _prepared(
    service: ServiceClient, run_id: str, display: ProgressDisplay
) -> PreparationHandle:
    route = f"/v1/runs/{service.session_id}/{run_id}"
    while True:
        service._work.check()
        reply = service._transport.request("GET", route)
        info = _info(service, reply.value, run_id)
        if info.kind != "preparation":
            raise ValueError("run is not a measurement-only preparation job")
        display.update(
            service,
            info,
            final=info.state in ("failed", "stopped")
            or (reply.status == 200 and info.state != "preparing"),
        )
        if info.state in ("failed", "stopped"):
            raise CreationError(info)
        if reply.status == 200 and info.state == "prepared":
            return PreparationHandle(info, service)
        service._work.pause()


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
    display = ProgressDisplay(progress)
    settings = service._setup(config, setup)
    frozen_labels = organization(name, {} if labels is None else labels)
    if type(wait_for_resources) is not bool:
        raise TypeError("wait_for_resources must be boolean")
    future = PreparationFuture(service.session_id, uuid4().hex)

    def create() -> PreparationHandle:
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
            kind="preparation",
        )
        _info(
            service,
            service._transport.request(
                "POST", f"/v1/runs/{service.session_id}/{future.run_id}", creation
            ).value,
            future.run_id,
        )
        return _prepared(service, future.run_id, display)

    service._work.submit(create, future=future)
    return future


def get_preparation(
    service: ServiceClient, run_id: str, *, progress: bool = False
) -> PreparationFuture:
    """Reattach read-only after disconnect, without recreating or replaying a job."""
    identifier(run_id, "preparation run ID")
    display = ProgressDisplay(progress)
    future = PreparationFuture(service.session_id, run_id)
    service._work.submit(lambda: _prepared(service, run_id, display), future=future)
    return future
