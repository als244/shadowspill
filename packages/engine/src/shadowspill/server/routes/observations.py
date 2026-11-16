"""Run metrics, reports and finalized profiler capture references."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.metrics import MetricAnnotation
from shadowspill_client.protocol.reports import ReportSettings
from shadowspill_client.protocol.service import CONTROL_MEDIA_TYPE

from ..nsight import missing_manifest, read_manifest
from ..records.requests import RunIdentity
from .runs import require_scope

if TYPE_CHECKING:
    from ..http import RunAccess, _Body, _Handler


def report(
    exchange: _Handler, identity: RunIdentity, run: RunAccess, body: _Body
) -> None:
    if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
        raise ProtocolError("report has no control media type")
    settings = body.record()
    if not isinstance(settings, ReportSettings):
        raise ProtocolError("report requires bounded settings")
    from ..reports import generate_report

    plans = None
    if exchange.server.jobs is not None:
        info = exchange.server.jobs.storage.catalog.get(identity)
        if info is None:
            raise KeyError("report run")
        require_scope(info, identity)
        plans = info.plans
    with exchange.server.artifacts(identity.owner_id) as store:
        exchange._reply(generate_report(run.records, store, settings, plans=plans))
    return


def metrics(
    exchange: _Handler, run: RunAccess, query: dict[str, list[str]], body: _Body
) -> None:
    if exchange.command == "GET":
        if set(query) - {"after", "through", "limit"}:
            raise ProtocolError("unknown metric query field")
        exchange._reply(
            run.records.metrics(
                after=int(query.get("after", ["0"])[0]),
                through=None if "through" not in query else int(query["through"][0]),
                limit=int(query.get("limit", ["100"])[0]),
            )
        )
        return
    if exchange.command == "POST" and not query:
        if exchange.headers.get("Content-Type") != CONTROL_MEDIA_TYPE:
            raise ProtocolError("caller metrics have no control media type")
        annotation = body.record()
        if not isinstance(annotation, MetricAnnotation):
            raise ProtocolError("caller metrics require an annotation")
        exchange._reply(run.records.log_metrics(annotation))
        return
    raise KeyError("route")


def nsys(
    exchange: _Handler, identity: RunIdentity, query: dict[str, list[str]], body: _Body
) -> None:
    if exchange.command != "GET" or set(query) != {"rank"} or body.length:
        raise ProtocolError("invalid capture route")
    rank = int(query["rank"][0])
    jobs = exchange.server.jobs
    if jobs is None:
        raise KeyError("capture routes are unavailable")
    info = jobs.storage.catalog.get(identity)
    if info is None:
        raise KeyError("capture run")
    require_scope(info, identity)
    with exchange.server.artifacts(identity.owner_id) as store:
        value = read_manifest(store, identity, rank)
        if value is None:
            raise KeyError("rank capture")
        if info.plans is not None and value.plans != info.plans:
            raise ValueError("capture differs from its source Plans")
        if value.status == "pending" and info.state not in ("preparing", "ready"):
            # A restart may recover a terminal run but only a pending record.
            # Report the missing finalization; never infer a usable report.
            value = missing_manifest(
                value, "Run ended without a finalized capture record."
            )
        exchange._reply(value)
