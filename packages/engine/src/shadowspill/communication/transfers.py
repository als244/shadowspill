"""Collectives inside compound fetch/evict operations, grouped by storage alias."""

from collections.abc import Mapping
from typing import Any

from .requests import CollectiveRequest


def transfer_collectives(
    bindings: Mapping[str, Any],
) -> dict[str, tuple[CollectiveRequest, ...]]:
    result = {}
    for kind in ("weight", "gradient"):
        for alias, shard in bindings.get(f"{kind}_shards", {}).items():
            requests = [
                CollectiveRequest(
                    "all_gather" if kind == "weight" else "reduce_scatter",
                    tuple(group),
                    shard["padded_bytes"],
                    shard["dtype"],
                    inplace=True,
                )
                for group in shard["groups"]
            ]
            requests.extend(
                CollectiveRequest(
                    "all_reduce", tuple(group), shard["shard_bytes"], shard["dtype"]
                )
                for group in shard.get("replica_groups", ())
            )
            result[alias] = tuple(requests)
    return result
