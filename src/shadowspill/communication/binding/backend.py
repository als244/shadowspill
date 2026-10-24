"""Collect one shared backend identity from the participating workers."""

from typing import Any

from ..backend import backend_setup
from ..requests import CollectiveRequest
from .groups import GroupBindings


def shared_backend_setup(
    bindings: GroupBindings | None,
    requests: tuple[CollectiveRequest, ...],
    *,
    device_ordinal: int | None,
) -> dict[str, Any]:
    if bindings is None:
        return {}
    local, failure = None, None
    try:
        groups = {
            ",".join(map(str, ranks)): bindings.for_members(ranks)
            for ranks in sorted({request.ranks for request in requests})
            if len(ranks) > 1 and bindings.logical_rank in ranks
        }
        if groups:
            local = backend_setup(groups, device_ordinal=device_ordinal)
    except Exception as error:
        failure = str(error)
    bindings.check(failure)
    rows = [row for row in bindings.gather(local) if row is not None]
    if not rows:
        return {}
    common = {key: value for key, value in rows[0].items() if key != "groups"}
    groups = {}
    for row in rows:
        if {key: value for key, value in row.items() if key != "groups"} != common:
            raise ValueError("workers have different communication backend setups")
        for name, value in row["groups"].items():
            if name in groups and groups[name] != value:
                raise ValueError("participants have different communicator options")
            groups[name] = value
    return {**common, "groups": dict(sorted(groups.items()))}
