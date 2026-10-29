"""Collect one shared backend identity from the participating workers."""

from typing import Any

from ..backend import backend_setup
from ..requests import CollectiveRequest
from .groups import GroupBindings

#: What names a node's own way onto the network, or what it logs: each node
#: keeps its own, and every other NCCL setting must agree across the workers.
LOCAL_SETTINGS = frozenset(
    {
        "NCCL_SOCKET_IFNAME",
        "NCCL_SOCKET_FAMILY",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_TC",
        "NCCL_IB_SL",
        "NCCL_IB_TIMEOUT",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_DEBUG_FILE",
    }
)


#: Variables naming where something is installed or written on a node, not
#: how the collectives run: each node's own.
LOCAL_SUFFIXES = ("_HOME", "_ROOT", "_PATH", "_DIR", "_FILE")


def is_local(name: str) -> bool:
    """Whether a setting is a node's own rather than the group's."""
    return name in LOCAL_SETTINGS or name.endswith(LOCAL_SUFFIXES)


def split_environment(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A worker's backend record as the part every worker must share and the
    part that is its own."""
    environment = row.get("environment", {})
    shared = {
        key: value for key, value in row.items() if key not in ("groups", "environment")
    }
    shared["environment"] = {k: v for k, v in environment.items() if not is_local(k)}
    return shared, {k: v for k, v in environment.items() if is_local(k)}


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
    common, _ = split_environment(rows[0])
    groups = {}
    local = []
    for row in rows:
        shared, own = split_environment(row)
        if shared != common:
            differing = sorted(
                key
                for key in set(shared) | set(common)
                if shared.get(key) != common.get(key)
            )
            detail = "; ".join(
                f"{key}: {common.get(key)!r} against {shared.get(key)!r}"
                for key in differing
            )
            raise ValueError(
                f"workers have different communication backend setups: {detail}"
            )
        local.append(own)
        for name, value in row["groups"].items():
            if name in groups and groups[name] != value:
                raise ValueError("participants have different communicator options")
            groups[name] = value
    return {
        **common,
        "groups": dict(sorted(groups.items())),
        "local_environment": local,
    }
