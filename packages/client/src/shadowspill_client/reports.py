"""Explicit recorded-evidence reports with separately requested downloads."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .protocol.reports import ReportDocument, ReportSettings
from .protocol.requests import identifier

if TYPE_CHECKING:
    from .service import ServiceClient


@dataclass(frozen=True)
class ReportHandle(ArtifactHandle):
    document: ReportDocument


def report(
    service: ServiceClient,
    session_id: str,
    run_id: str,
    *,
    format: Literal["html", "json"] = "html",
    after: int = 0,
    through: int | None = None,
    max_records: int = 10000,
    max_bytes: int = 64 << 20,
) -> CallFuture[ReportHandle]:
    identifier(session_id, "report session")
    identifier(run_id, "report run")
    settings = ReportSettings(format, after, through, max_records, max_bytes)

    def generate() -> ReportHandle:
        value = service._request(
            "POST", f"/v1/runs/{session_id}/{run_id}/report", ReportDocument, settings
        )
        if (
            (value.snapshot.session_id, value.snapshot.run_id) != (session_id, run_id)
            or value.settings != replace(settings, through=value.snapshot.through)
            or (through is not None and through != value.snapshot.through)
        ):
            raise ValueError("report reply differs from the requested run or snapshot")
        return ReportHandle(value.reference, service, value)

    return service._work.submit(generate)
