"""Read scalar history without a live writer, model imports or implicit downloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._async import CallFuture
from .protocol.codec import ProtocolError
from .protocol.metrics import MetricPage, metric_cursor
from .protocol.requests import identifier

if TYPE_CHECKING:
    from .service import ServiceClient


class MetricQueries:
    def __init__(self, service: ServiceClient, session_id: str, run_id: str) -> None:
        self._service = service
        self.session_id = identifier(session_id, "metric session")
        self.run_id = identifier(run_id, "metric run")

    def query(
        self, *, after: int = 0, through: int | None = None, limit: int = 100
    ) -> CallFuture[MetricPage]:
        """Read completion order; pin ``through`` when traversing a fixed snapshot."""
        metric_cursor(after, "metric cursor")
        if through is not None and metric_cursor(through, "metric snapshot") < after:
            raise ValueError("metric cursor exceeds its snapshot")
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("metric query limit must be 1..128")
        route = (
            f"/v1/runs/{self.session_id}/{self.run_id}/metrics"
            f"?after={after}&limit={limit}"
        )
        if through is not None:
            route += f"&through={through}"

        def read() -> MetricPage:
            page = self._service._request("GET", route, MetricPage)
            if (
                page.session_id != self.session_id
                or page.run_id != self.run_id
                or page.after != after
                or len(page.records) > limit
                or (through is not None and page.through != through)
            ):
                raise ProtocolError(
                    "metric reply differs from its requested scope or snapshot"
                )
            return page

        return self._service._work.submit(read)
