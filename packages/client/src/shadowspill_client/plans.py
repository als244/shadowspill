"""Read saved metadata independently of the training connection's mutable state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .protocol.plans import (
    PlanReportDocument,
    PlanReportRequest,
    PlanSummaryPage,
    SavedPlans,
    plan_name,
    summary_id,
)
from .protocol.requests import identifier

if TYPE_CHECKING:
    from .service import ServiceClient


@dataclass(frozen=True)
class PlansHandle(ArtifactHandle):
    _run_id: str | None = field(default=None, repr=False, compare=False)

    def report(self, *, max_bytes: int = 64 << 20) -> CallFuture[ArtifactHandle]:
        """Render all saved candidates in one standalone, offline HTML report."""
        if self._run_id is None:
            raise ValueError("Plan reports require a training or saved-name handle")
        request = PlanReportRequest(
            self._service.session_id, self._run_id, self.reference, max_bytes
        )
        route = f"/v1/runs/{self._service.session_id}/{self._run_id}/plans/report"

        def render() -> ArtifactHandle:
            document = self._service._request(
                "POST", route, PlanReportDocument, request
            )
            if document.request != request:
                raise ValueError("Plan report reply differs from requested Plans")
            return ArtifactHandle(document.reference, self._service)

        return self._service._work.submit(render)

    def save(self, *, name: str) -> CallFuture[PlansHandle]:
        """Name this run's existing Plans on the server; no weights or download."""
        if self._run_id is None:
            raise ValueError("saving Plans requires a training or saved-name handle")
        saved = SavedPlans(self._service.session_id, self._run_id, name, self.reference)
        route = f"/v1/runs/{saved.session_id}/{saved.run_id}/plans"

        def persist() -> PlansHandle:
            result = self._service._request("POST", route, SavedPlans, saved)
            if result != saved:
                raise ValueError("saved Plan reply differs from the requested name")
            return PlansHandle(result.plans, self._service, result.run_id)

        return self._service._work.submit(persist)

    def summary(self, *, page: int = 0) -> CallFuture[PlanSummaryPage]:
        """Read one immutable metadata page; full bundle download stays explicit."""
        route = "/v1/artifacts/" + summary_id(self.reference, page)

        def inspect() -> PlanSummaryPage:
            result = self._service._request("GET", route, PlanSummaryPage)
            if result.plans != self.reference or result.page != page:
                raise ValueError("saved summary belongs to different Plans/page")
            return result

        return self._service._work.submit(inspect)


def get_plans(
    service: ServiceClient, run_id: str, *, name: str
) -> CallFuture[PlansHandle]:
    identifier(run_id, "Plan run ID")
    plan_name(name)
    route = f"/v1/runs/{service.session_id}/{run_id}/plans?" + urlencode({"name": name})

    def retrieve() -> PlansHandle:
        saved = service._request("GET", route, SavedPlans)
        if (saved.session_id, saved.run_id, saved.name) != (
            service.session_id,
            run_id,
            name,
        ):
            raise ValueError("saved Plan reply belongs to another run or name")
        return PlansHandle(saved.plans, service, run_id)

    return service._work.submit(retrieve)
