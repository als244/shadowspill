"""Remote creation/reattachment; no build, planner or worker implementation here."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .checkpoints import CheckpointHandle
from .protocol.codec import MAX_METADATA_BYTES, decode, digest, encode
from .protocol.creation import TrainingCreation
from .protocol.plans import PlanSummaryPage, summary_id
from .protocol.preview import TrainingPreview
from .protocol.requests import ArtifactRef, WriterLease, identifier
from .protocol.runs import RunInfo
from .protocol.setup import TrainingSetup
from .protocol.training import ResolvedTraining
from .training import TrainingClient

if TYPE_CHECKING:
    from .service import ServiceClient


class CreationError(RuntimeError):
    def __init__(self, info: RunInfo) -> None:
        self.info = info
        super().__init__(f"training run {info.run_id} is {info.state}: {info.error}")


class TrainingFuture(CallFuture[TrainingClient]):
    def __init__(self, session_id: str, run_id: str) -> None:
        super().__init__()
        self.session_id, self.run_id = session_id, run_id


def _info(service: ServiceClient, value: object, run_id: str) -> RunInfo:
    if not isinstance(value, RunInfo) or (
        value.owner_id != service.info.owner_id
        or value.session_id != service.session_id
        or value.run_id != run_id
    ):
        raise ValueError("creation lookup crossed authenticated run scope")
    return value


def _ready(service: ServiceClient, run_id: str, *, read_only: bool) -> RunInfo:
    route = f"/v1/runs/{service.session_id}/{run_id}"
    while True:
        service._work.check()
        reply = service._transport.request("GET", route)
        info = _info(service, reply.value, run_id)
        if info.kind != "training":
            raise ValueError("measurement-only preparation has no training client")
        if read_only or (reply.status == 200 and info.state == "ready"):
            return info
        if info.state in ("failed", "closed", "stopped"):
            raise CreationError(info)
        service._work.pause()


def _attach(service: ServiceClient, info: RunInfo) -> WriterLease:
    writer_id = uuid4().hex
    route = f"/v1/runs/{info.session_id}/{info.run_id}/writers/{writer_id}"
    while True:
        service._work.check()
        reply = service._transport.request("POST", route)
        writer = reply.value
        if not isinstance(writer, WriterLease) or (
            writer.run_id != info.run_id
            or writer.writer_id != writer_id
            or writer.server_incarnation != service.info.server_incarnation
        ):
            raise ValueError("writer attachment returned a different identity")
        if reply.status == 200:
            return writer
        if reply.status != 202:
            raise ValueError("unknown writer attachment progress")
        # A confirmed 202 continues this exact attachment. An exchange failure
        # propagates; it does not retry or acquire a replacement writer silently.
        service._work.pause()


def _config(service: ServiceClient, reference: ArtifactRef) -> ResolvedTraining:
    if reference.size_bytes > MAX_METADATA_BYTES:
        raise ValueError("saved configuration exceeds its control metadata bound")
    with BytesIO() as stream:
        service._transport.download("/v1/artifacts/" + reference.id, reference, stream)
        value = decode(stream.getvalue())
    if not isinstance(value, ResolvedTraining):
        raise ValueError("run artifact is not a resolved configuration")
    return value


def get_training_client(
    service: ServiceClient, run_id: str, *, read_only: bool = False
) -> TrainingFuture:
    identifier(run_id, "run ID")
    if type(read_only) is not bool:
        raise TypeError("read_only must be boolean")
    future = TrainingFuture(service.session_id, run_id)

    def attach() -> TrainingClient:
        info = _ready(service, run_id, read_only=read_only)
        config = _config(service, info.configuration)
        writer = None if read_only else _attach(service, info)
        return TrainingClient(service, info, config, writer)

    service._work.submit(attach, future=future)
    return future


def _resume_config(
    service: ServiceClient, plans: ArtifactRef, checkpoint: ArtifactRef
) -> ResolvedTraining:
    manifest = CheckpointHandle(checkpoint, service)._manifest()[0]
    if manifest.info.plans == plans:
        # A copied native checkpoint already pins its exact resolved settings.
        # It need not carry a generated summary/cache page from the old server.
        reference = manifest.info.configuration
        del manifest
        return _config(service, reference)
    del manifest
    # Changed ownership uses the target Plans' settings, not the source layout.
    page = service._request(
        "GET", "/v1/artifacts/" + summary_id(plans, 0), PlanSummaryPage
    )
    if page.plans != plans or page.page != 0:
        raise ValueError("saved configuration summary belongs to different Plans")
    resolved = service._request(
        "GET",
        "/v1/artifacts/configuration." + page.configuration_sha256,
        ResolvedTraining,
    )
    if digest(resolved) != page.configuration_sha256:
        raise ValueError("saved configuration differs from its Plans summary")
    return resolved


def create_training_client(
    service: ServiceClient,
    *,
    config: ResolvedTraining | None = None,
    plans: ArtifactRef | ArtifactHandle | None = None,
    checkpoint: ArtifactRef | ArtifactHandle | None = None,
    wait_for_resources: bool = False,
    name: str | None = None,
    labels: Mapping[str, str] | None = None,
    **setup: Any,
) -> TrainingFuture:
    plan_ref = plans.reference if isinstance(plans, ArtifactHandle) else plans
    checkpoint_ref = (
        checkpoint.reference if isinstance(checkpoint, ArtifactHandle) else checkpoint
    )
    # Freeze organization/control settings before the asynchronous description.
    from .protocol.runs import organization

    frozen_labels = organization(name, {} if labels is None else labels)
    if type(wait_for_resources) is not bool:
        raise TypeError("wait_for_resources must be boolean")
    for value in (plan_ref, checkpoint_ref):
        if value is not None and not isinstance(value, ArtifactRef):
            raise TypeError("saved Plans/checkpoint require server artifact references")
    settings = (
        None
        if config is None
        and not setup
        and plan_ref is not None
        and checkpoint_ref is not None
        else service._setup(config, setup)
    )
    future = TrainingFuture(service.session_id, uuid4().hex)

    def create() -> TrainingClient:
        if settings is None:
            assert plan_ref is not None and checkpoint_ref is not None
            resolved = _resume_config(service, plan_ref, checkpoint_ref)
        elif isinstance(settings, TrainingSetup):
            resolved = service._request(
                "POST", "/v1/training/describe", TrainingPreview, settings
            ).config
        else:
            resolved = settings
        creation = TrainingCreation(
            resolved,
            plans=plan_ref,
            checkpoint=checkpoint_ref,
            wait_for_resources=wait_for_resources,
            name=name,
            labels=frozen_labels,
        )
        encode(creation)
        route = f"/v1/runs/{service.session_id}/{future.run_id}"
        info = _info(
            service,
            service._transport.request("POST", route, creation).value,
            future.run_id,
        )
        if info.state in ("failed", "closed", "stopped"):
            raise CreationError(info)
        info = _ready(service, future.run_id, read_only=False)
        return TrainingClient(service, info, resolved, _attach(service, info))

    service._work.submit(create, future=future)
    return future
