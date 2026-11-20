"""Inspect per-rank capture status without retaining a live training writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._async import CallFuture
from .artifacts import ArtifactHandle
from .protocol.captures import NsightManifest, rank_name
from .protocol.requests import identifier

if TYPE_CHECKING:
    from .service import ServiceClient


@dataclass(frozen=True)
class NsightCapture:
    """One worker's Nsight Systems capture: its status and report.

    `TrainingClient.nsys` and `ServiceClient.get_nsys` return it. Its report
    stays pending until the worker exits.

    Attributes
    ----------
    record : NsightManifest
        The capture's record: its ``status``, the ``window`` of updates it
        covers and its ``report``.
    """

    record: NsightManifest
    _service: ServiceClient = field(repr=False, compare=False)

    @property
    def report(self) -> ArtifactHandle | None:
        """The native report file, or None when the capture has none.

        A ready capture has one, and a failed capture may keep a partial one.
        """
        return (
            None
            if self.record.report is None
            else self._service.artifact(self.record.report)
        )


def get_nsys(
    service: ServiceClient, run_id: str, *, rank: int
) -> CallFuture[NsightCapture]:
    identifier(run_id, "capture run ID")
    rank_name(rank)
    route = f"/v1/runs/{service.session_id}/{run_id}/nsys?rank={rank}"

    def retrieve() -> NsightCapture:
        value = service._request("GET", route, NsightManifest)
        if (value.session_id, value.run_id, value.rank) != (
            service.session_id,
            run_id,
            rank,
        ):
            raise ValueError("capture reply belongs to another run or rank")
        return NsightCapture(value, service)

    return service._work.submit(retrieve)
