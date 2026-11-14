"""Compose one calibration request from a caller's compatible Programs."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shadowspill import Program


def communication_probes(
    programs: Iterable["Program"],
) -> tuple["Program", list[dict[str, Any]]]:
    """Union exact collective geometries; keep the first Program as context.

    This consumes Programs one at a time and performs no measurement. Callers
    measure the union in one worker group, then independently measure each
    candidate's task/communication overlap. Different setup or parallelism
    requires a separate calibration. Ordered ranks and buffer aliasing remain
    part of each distinct collective request.
    """
    from shadowspill.communication.requests import (
        CollectiveRequest,
        collective_requests,
    )

    first = None
    requests: set[CollectiveRequest] = set()
    for program in programs:
        if first is None:
            first = program
        elif program.parallelism != first.parallelism:
            raise ValueError(
                "communication candidates must declare the same parallelism"
            )
        elif program.setup != first.setup:
            raise ValueError("communication candidates must have the same build setup")
        requests.update(collective_requests(program, None))
    if first is None:
        raise ValueError("communication calibration requires at least one Program")
    return first, [request.to_dict() for request in sorted(requests)]
