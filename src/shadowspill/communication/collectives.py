"""CPU collective costs with explicit full-tensor geometry and provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import nonnegative_int, plain, seconds

if TYPE_CHECKING:
    from .info import CommunicationInfo

_OPERATIONS = {"all_gather", "reduce_scatter", "all_reduce"}


def _inplace(row: Mapping[str, Any]) -> bool:
    value = row.get("inplace")
    if value is None:
        value = row["operation"] == "all_reduce"
    if (
        type(value) is not bool
        or (row["operation"] == "all_reduce" and not value)
        or (row["operation"] == "reduce_scatter" and value)
    ):
        raise ValueError("unsupported collective buffer aliasing")
    return value


@dataclass(frozen=True)
class CollectiveCost:
    duration_seconds: float
    transfer_bytes: int
    source: str
    assumptions: tuple[str, ...] = ()


def validate_measurement(row: Mapping[str, Any]) -> None:
    """Validate a plain observation; setup/ranks are part of its compatibility."""
    if row.get("operation") not in _OPERATIONS:
        raise ValueError("collective measurement has an unsupported operation")
    _inplace(row)
    ranks = row.get("ranks", ())
    if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks):
        raise ValueError("collective measurement requires ordered logical ranks")
    if len(set(ranks)) != len(ranks):
        raise ValueError("collective measurement has duplicate logical ranks")
    if "tensor_bytes" not in row or "duration_seconds" not in row:
        raise ValueError("collective measurement requires size and duration")
    nonnegative_int(row["tensor_bytes"], "tensor_bytes")
    if row["operation"] != "all_reduce" and row["tensor_bytes"] % len(ranks):
        raise ValueError(
            "gather/scatter measurement must include padding to group size"
        )
    if not isinstance(row.get("dtype"), str) or not row["dtype"]:
        raise ValueError("collective measurement requires dtype")
    seconds(row["duration_seconds"], "duration_seconds")
    if not isinstance(row.get("setup"), Mapping) or not row["setup"]:
        raise ValueError("collective measurement requires hardware/runtime setup")
    topology = row.get("topology")
    if not isinstance(topology, Mapping):
        raise ValueError("collective measurement requires topology")
    participants = [rank for members in topology.values() for rank in members]
    if len(set(participants)) != len(participants):
        raise ValueError("collective topology assigns one rank to multiple nodes")
    if len(ranks) > 1 and not set(ranks) <= set(participants):
        raise ValueError("collective cost requires topology for every participant")


def collective_cost(
    info: CommunicationInfo,
    *,
    operation: str,
    ranks: tuple[int, ...],
    tensor_bytes: int,
    dtype: str,
    setup: Mapping[str, Any],
    inplace: bool | None = None,
) -> CollectiveCost:
    """Prefer exact observations; otherwise estimate a synchronous flat ring.

    ``tensor_bytes`` is the padded gathered output/scatter input, or each
    rank's complete all-reduce tensor. Bandwidth is per participant/direction,
    not the sum of sent and received bytes. Costs never include host copies.
    """
    request = dict(
        operation=operation,
        ranks=ranks,
        tensor_bytes=tensor_bytes,
        dtype=dtype,
        setup=setup,
        topology=info.topology,
        duration_seconds=0.0,
        inplace=inplace,
    )
    validate_measurement(request)
    count = len(ranks)
    if count == 1:
        return CollectiveCost(0.0, 0, "identity")
    factor = 2 if operation == "all_reduce" else 1
    # Integer upper bound also permits an unpadded all-reduce tensor.
    chunk = (tensor_bytes + count - 1) // count
    traffic = factor * (count - 1) * chunk
    fields = ("operation", "ranks", "tensor_bytes", "dtype", "setup", "topology")
    for row in reversed(info.measurements):
        if _inplace(row) == _inplace(request) and all(
            plain(row[field]) == plain(request[field]) for field in fields
        ):
            return CollectiveCost(
                row["duration_seconds"],
                traffic,
                "measurement",
                (
                    "transfer_bytes uses ring-equivalent per-rank traffic; "
                    "the measured backend algorithm is not inferred",
                ),
            )
    nodes = {rank: node for node, members in info.topology.items() for rank in members}
    scopes = {
        "scale_up" if nodes[left] == nodes[right] else "scale_out"
        for left, right in zip(ranks, (*ranks[1:], ranks[0]), strict=True)
    }
    step_seconds = []
    for scope in sorted(scopes):
        bandwidth = getattr(info, f"{scope}_bytes_per_second")
        latency = getattr(info, f"{scope}_latency_seconds")
        if bandwidth is None or latency is None:
            raise ValueError(f"collective estimation needs {scope} rate and latency")
        step_seconds.append(latency + chunk / bandwidth)
    return CollectiveCost(
        factor * (count - 1) * max(step_seconds),
        traffic,
        "ring",
        (
            "flat ring in declared logical group order",
            "effective per-participant directional rates; "
            "no additional shared-link contention",
        ),
    )
