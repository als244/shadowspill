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
    """A run's saved Plan bundle.

    `TrainingClient.plans`, `ServiceClient.get_plans`, a planning job and
    `save` give one. As an `ArtifactHandle`, its `ArtifactHandle.download`
    exports the full bundle. The other calls read metadata the service saved
    with the Plans: they need no workers, and work after the run's workers
    close or the service restarts.

    Attributes
    ----------
    reference : ArtifactRef
        The Plan bundle.
    """

    _run_id: str | None = field(default=None, repr=False, compare=False)

    def report(self, *, max_bytes: int = 64 << 20) -> CallFuture[ArtifactHandle]:
        """Render the saved candidates and worker Plans into one HTML report.

        The page is standalone and works offline. The service renders it once
        for these Plans and ``max_bytes`` and stores it; repeating the call
        returns the stored report.

        Parameters
        ----------
        max_bytes : int, default 64 << 20
            Largest report size, in bytes (64 MiB by default), up to 1 GiB.
            Rendering a larger report fails.

        Returns
        -------
        CallFuture of ArtifactHandle
            The stored HTML report.

        Raises
        ------
        ValueError
            If ``max_bytes`` is below one or above 1 GiB.
        """
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
        """Give these Plans a name within their run.

        The service records the name only: nothing is copied or downloaded,
        and no training request is queued. Saving the same name again changes
        nothing. `ServiceClient.get_plans` finds them by run and
        name from any connection to the same session. Plan names are separate
        from checkpoint names.

        Parameters
        ----------
        name : str
            The name, at most 256 characters, without surrounding spaces.

        Returns
        -------
        CallFuture of PlansHandle
            The named Plans.

        Raises
        ------
        ValueError
            If ``name`` is empty, has surrounding spaces, exceeds 256
            characters or cannot be encoded as UTF-8.
        """
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
        """Read one page of the metadata saved with these Plans.

        Pages are numbered from 0; follow `PlanSummaryPage.next_page` until it
        is None.

        Parameters
        ----------
        page : int, default 0
            The page number.

        Returns
        -------
        CallFuture of PlanSummaryPage
            The page.

        Raises
        ------
        ValueError
            If ``page`` is negative.
        """
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
