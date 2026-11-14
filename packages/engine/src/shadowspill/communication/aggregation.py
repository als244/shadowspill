"""Conservative shared host rates with every worker's observations preserved."""

from .binding.groups import GroupBindings
from .info import CommunicationInfo


def combine_host(
    local: CommunicationInfo, bindings: GroupBindings | None
) -> CommunicationInfo:
    if bindings is None:
        return local
    rows = bindings.gather({"logical_rank": bindings.logical_rank, **local.to_dict()})
    values = rows[0].copy()
    del values["logical_rank"]
    for direction in ("fetch", "evict"):
        for suffix in ("bytes_per_second", "concurrent_bytes_per_second"):
            name = f"{direction}_{suffix}"
            values[name] = min(row[name] for row in rows)
        name = f"{direction}_latency_seconds"
        values[name] = max(row[name] for row in rows)
    values["topology"] = bindings.topology
    values["metadata"] = {
        "source": "simultaneous per-worker spill transfers",
        "spill_kind": local.metadata.get("spill_kind", "pinned_host"),
        "scope": "minimum observed rates and maximum fitted latency across workers",
        "per_rank": sorted(rows, key=lambda row: row["logical_rank"]),
    }
    return CommunicationInfo.from_dict(values)
