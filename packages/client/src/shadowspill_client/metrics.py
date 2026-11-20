"""Read scalar history without a live writer, model imports or implicit downloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from ._async import CallFuture
from .protocol.codec import MAX_METADATA_BYTES, ProtocolError, digest, encode
from .protocol.metrics import (
    MetricAnnotation,
    MetricPage,
    MetricRecord,
    metric_cursor,
)
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
    """Reads of a run's metric journal.

    `TrainingClient.metrics` is one. The journal holds one record per
    finished request, of any operation, in completion order, and the values
    logged with `TrainingClient.log_metrics`. Reads need no workers and work
    after the workers close or the service restarts; inputs and tensor
    outputs are never read.

    Attributes
    ----------
    session_id : str
        Session that owns the run.
    run_id : str
        The run.
    """

    def __init__(self, service: ServiceClient, session_id: str, run_id: str) -> None:
        self._service = service
        self.session_id = identifier(session_id, "metric session")
        self.run_id = identifier(run_id, "metric run")

    def _log(
        self,
        values: Mapping[str, float | int],
        *,
        request_id: str,
        metric_id: str | None = None,
    ) -> CallFuture[MetricRecord]:
        annotation = MetricAnnotation(
            request_id, values, "content" if metric_id is None else metric_id
        )
        if metric_id is None:
            # Explicitly repeating identical content after a lost reply is safe.
            # There is no automatic network retry or training request involved.
            annotation = replace(annotation, metric_id=digest(annotation))
        encode(annotation)

        def append() -> MetricRecord:
            row = self._service._request(
                "POST",
                f"/v1/runs/{self.session_id}/{self.run_id}/metrics",
                MetricRecord,
                annotation,
            )
            if (
                row.source != "caller"
                or row.identity.run_id != self.run_id
                or row.identity.request_id != annotation.request_id
                or row.annotation_id != annotation.metric_id
                or row.metrics != annotation.values
            ):
                raise ProtocolError("caller metric reply differs from its annotation")
            return row

        return self._service._work.submit(append)

    def query(
        self, *, after: int = 0, through: int | None = None, limit: int = 100
    ) -> CallFuture[MetricPage]:
        """Read one page of the run's metric journal.

        Records come in journal position order, which is completion order.
        To read a fixed snapshot, keep the first page's `MetricPage.through`
        as ``through`` and pass each page's `MetricPage.next_cursor` as
        ``after`` until it is None. To poll for later records, start after the
        finished snapshot's ``through``.

        Parameters
        ----------
        after : int, default 0
            Journal position to start after.
        through : int, optional
            Last journal position to read. By default the journal's current
            end.
        limit : int, default 100
            Maximum number of records in the page, from 1 to 128. The service may return
            fewer to keep the page within its metadata size bound.

        Returns
        -------
        CallFuture of MetricPage
            The page.

        Raises
        ------
        ValueError
            If ``after`` or ``through`` is not a nonnegative int below 2**63,
            ``through`` is below ``after``, or ``limit`` is outside 1 to 128.
        """
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
        """Write one fixed snapshot of the metric journal to a new JSON Lines file.

        Each line is one `MetricRecord` as an ordinary JSON object, keyed by
        its field names, in journal order. A snapshot without records writes
        an empty file. Pages are read one at a time, and records finished
        later are not added. A failure removes the partial file.

        Parameters
        ----------
        path : str or pathlib.Path
            The local file to create.
        after : int, default 0
            Journal position to start after.
        through : int, optional
            Last journal position to write. By default the journal's end when
            the first page is read.
        limit : int, default 100
            Records per page read, from 1 to 128.

        Returns
        -------
        CallFuture of pathlib.Path
            ``path``, once the file is complete. It fails with
            `FileExistsError` if ``path`` exists.

        Raises
        ------
        ValueError
            If ``after``, ``through`` or ``limit`` is invalid, as in `query`.
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
                        while True:
                            self._service._work.check()
                            for row in page.records:
                                line = json.dumps(row.to_dict(), allow_nan=False)
                                stream.write(line.encode() + b"\n")
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
