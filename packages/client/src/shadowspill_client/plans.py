"""Read saved metadata independently of the training connection's mutable state."""

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .protocol.plans import PlanSummaryPage, summary_id


class PlansHandle(ArtifactHandle):
    def summary(self, *, page: int = 0) -> CallFuture[PlanSummaryPage]:
        """Read one immutable metadata page; full bundle download stays explicit."""
        route = "/v1/artifacts/" + summary_id(self.reference, page)

        def inspect() -> PlanSummaryPage:
            result = self._service._request("GET", route, PlanSummaryPage)
            if result.plans != self.reference or result.page != page:
                raise ValueError("saved summary belongs to different Plans/page")
            return result

        return self._service._work.submit(inspect)
