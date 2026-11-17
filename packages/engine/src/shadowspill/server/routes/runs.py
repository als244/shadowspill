"""Run creation, lifecycle inspection and names for saved Plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shadowspill_client.protocol.codec import (
    MAX_METADATA_BYTES,
    ProtocolError,
)
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.plans import (
    PlanReportDocument,
    PlanReportRequest,
    SavedPlans,
)
from shadowspill_client.protocol.progress import ProgressPage, ProgressRequest
from shadowspill_client.protocol.runs import RunInfo, RunPage
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE

from ..creation_logs import archived_chunks
from ..creation_records import CreationRecord
from ..plan_reports import generate_plan_report
from ..records.artifact_index import ArtifactAlias
from ..records.requests import RunIdentity
from ..records.run_index import RUN_BYTES

if TYPE_CHECKING:
    from ..http import _Body, _Handler


def creation(
    exchange: _Handler,
    owner: str,
    parts: tuple[str, ...],
    query: dict[str, list[str]],
    body: _Body,
) -> None:
    jobs = exchange.server.jobs
    if jobs is None:
        raise KeyError("creation routes are unavailable")
    catalog = jobs.storage.catalog
    if len(parts) == 3 and exchange.command == "GET":
        if set(query) - {"after", "limit"}:
            raise ProtocolError("unknown run-list query")
        limit = int(query.get("limit", ["32"])[0])
        if not 1 <= limit <= 128:
            raise ProtocolError("run page limit must be in 1..128")
        limit = min(limit, MAX_METADATA_BYTES // RUN_BYTES)
        rows = catalog.page(
            owner, parts[2], after=query.get("after", [""])[0], limit=limit
        )
        if any(row.owner_id != owner or row.session_id != parts[2] for row in rows):
            raise PermissionError("run page crossed authenticated scope")
        cursor = rows[-1].run_id if len(rows) == limit else None
        # Full lifecycle rows include failure/provenance headroom. Return
        # fewer rows when needed to fit the common control envelope.
        exchange._reply(RunPage(rows, cursor))
        return
    if len(parts) != 4 or query:
        raise KeyError("route")
    identity = RunIdentity(owner, parts[2], parts[3])
    if exchange.command == "POST":
        if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
            raise ProtocolError("creation has no control media type")
        creation = body.record()
        if not isinstance(creation, TrainingCreation):
            raise ProtocolError("creation requires resolved training choices")
        record = CreationRecord.encode(creation)

        def publish() -> None:
            with jobs.storage.artifacts(owner) as artifacts:
                record.publish(artifacts)

        jobs.create(
            identity,
            request_sha256=record.reference.sha256,
            configuration=record.configuration,
            name=creation.name,
            labels=creation.labels,
            kind=creation.kind,
            publish=publish,
        )
    elif exchange.command != "GET":
        raise KeyError("route")
    info = catalog.get(identity)
    if info is None:
        raise KeyError("run")
    reply(exchange, info, identity)


def plans(
    exchange: _Handler, identity: RunIdentity, query: dict[str, list[str]], body: _Body
) -> None:
    jobs = exchange.server.jobs
    if jobs is None:
        raise KeyError("saved Plan routes are unavailable")
    info = jobs.storage.catalog.get(identity)
    if info is None or info.plans is None:
        raise KeyError("run Plans")
    require_scope(info, identity)
    with exchange.server.artifacts(identity.owner_id) as store:
        if store.owner_id != identity.owner_id:
            raise PermissionError("Plan lookup crossed owner scope")
        if exchange.command == "POST" and not query:
            if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
                raise ProtocolError("Plan name has no control media type")
            saved = body.record()
            if not isinstance(saved, SavedPlans) or (
                saved.session_id,
                saved.run_id,
                saved.plans,
            ) != (identity.session_id, identity.run_id, info.plans):
                raise ProtocolError("saved Plans differ from their source run")
            store.name(
                saved.plans,
                ArtifactAlias(
                    saved.session_id,
                    saved.run_id,
                    saved.name,
                    saved.plans.id,
                    "plans",
                ),
            )
        elif exchange.command == "GET" and set(query) == {"name"}:
            name = query["name"][0]
            reference = store.resolve_alias(
                identity.session_id, identity.run_id, name, kind="plans"
            )
            if reference is None:
                raise KeyError("Plan name")
            if reference != info.plans:
                raise ValueError("saved Plans differ from their source run")
            saved = SavedPlans(identity.session_id, identity.run_id, name, reference)
        else:
            raise ProtocolError("invalid saved Plan route")
        exchange._reply(saved)


def require_scope(info: RunInfo, identity: RunIdentity) -> None:
    if (info.owner_id, info.session_id, info.run_id) != (
        identity.owner_id,
        identity.session_id,
        identity.run_id,
    ):
        raise PermissionError("run catalog crossed authenticated scope")


def progress(
    exchange: _Handler, identity: RunIdentity, query: dict[str, list[str]], body: _Body
) -> None:
    jobs = exchange.server.jobs
    if jobs is None:
        raise KeyError("creation progress is unavailable")
    if exchange.command != "POST" or query:
        raise ProtocolError("invalid creation progress route")
    if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
        raise ProtocolError("creation progress has no control media type")
    request = body.record()
    if not isinstance(request, ProgressRequest):
        raise ProtocolError("creation progress requires log cursors")
    info = jobs.storage.catalog.get(identity)
    if info is None:
        raise KeyError("run")
    require_scope(info, identity)
    live = jobs.logs(identity, request)
    with exchange.server.artifacts(identity.owner_id) as store:
        chunks = archived_chunks(store, identity, request, live)
    exchange._reply(ProgressPage(identity.session_id, identity.run_id, chunks))


def plan_report(
    exchange: _Handler, identity: RunIdentity, query: dict[str, list[str]], body: _Body
) -> None:
    jobs = exchange.server.jobs
    if jobs is None:
        raise KeyError("saved Plan routes are unavailable")
    if exchange.command != "POST" or query:
        raise ProtocolError("invalid Plan report route")
    if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
        raise ProtocolError("Plan report has no control media type")
    request = body.record()
    info = jobs.storage.catalog.get(identity)
    if info is None or info.plans is None:
        raise KeyError("run Plans")
    require_scope(info, identity)
    if not isinstance(request, PlanReportRequest) or (
        request.session_id,
        request.run_id,
        request.plans,
    ) != (identity.session_id, identity.run_id, info.plans):
        raise ProtocolError("Plan report differs from its source run")
    with exchange.server.artifacts(identity.owner_id) as store:
        exchange._reply(
            PlanReportDocument(request, generate_plan_report(store, identity, request))
        )


def reply(exchange: _Handler, info: RunInfo, identity: RunIdentity) -> None:
    require_scope(info, identity)
    assert exchange.server.jobs is not None
    pending = (
        info.state == "preparing"
        or (info.state == "ready" and not exchange.server.jobs.available(identity))
        or (
            info.state in ("closed", "failed", "stopped", "prepared")
            and not exchange.server.jobs.retired(identity)
        )
    )
    exchange._reply(info, 202 if pending else 200)
