"""Read scalar history without a live writer, model imports or implicit downloads."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._async import CallFuture
from .protocol.codec import MAX_METADATA_BYTES, ProtocolError, encode
from .protocol.metrics import MetricPage, MetricSnapshot, metric_cursor
from .protocol.requests import identifier

if TYPE_CHECKING:
    from .service import ServiceClient


def _query(after: int, through: int | None, limit: int) -> None:
    metric_cursor(after, "metric cursor")
    if through is not None and metric_cursor(through, "metric snapshot") < after:
        raise ValueError("metric cursor exceeds its snapshot")
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("metric query limit must be 1..128")


class MetricQueries:
    def __init__(self, service: ServiceClient, session_id: str, run_id: str) -> None:
        self._service = service
        self.session_id = identifier(session_id, "metric session")
        self.run_id = identifier(run_id, "metric run")

    def query(
        self, *, after: int = 0, through: int | None = None, limit: int = 100
    ) -> CallFuture[MetricPage]:
        """Read completion order; pin ``through`` when traversing a fixed snapshot."""
        _query(after, through, limit)
        return self._service._work.submit(lambda: self._read(after, through, limit))

    def _read(self, after: int, through: int | None, limit: int) -> MetricPage:
        route = (
            f"/v1/runs/{self.session_id}/{self.run_id}/metrics"
            f"?after={after}&limit={limit}"
        )
        if through is not None:
            route += f"&through={through}"

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

    def export_jsonl(
        self,
        path: str | Path,
        *,
        after: int = 0,
        through: int | None = None,
        limit: int = 100,
    ) -> CallFuture[Path]:
        """Stream one explicit snapshot into a new file, with a portable header.

        Each JSON line is a bounded codec record: MetricSnapshot followed by
        MetricRecords. The range stays fixed while later results arrive. Existing
        paths are preserved; a failure removes only this call's incomplete file.
        """
        _query(after, through, limit)
        destination = Path(path)

        def transfer() -> Path:
            # Same metadata allowance as rank-directory downloads; at most one
            # page plus one encoded row is live, independent of history length.
            with self._service._memory.reserve(16 * MAX_METADATA_BYTES):
                stream = destination.open("xb")
                try:
                    with stream:
                        page = self._read(after, through, limit)
                        stream.write(
                            encode(
                                MetricSnapshot(
                                    self.session_id, self.run_id, after, page.through
                                )
                            )
                            + b"\n"
                        )
                        while True:
                            self._service._work.check()
                            for row in page.records:
                                stream.write(encode(row) + b"\n")
                            if page.next_cursor is None:
                                break
                            cursor, snapshot = page.next_cursor, page.through
                            del page
                            page = self._read(cursor, snapshot, limit)
                except BaseException:
                    destination.unlink()
                    raise
            return destination

        return self._service._work.submit(transfer)
