"""Resolve communication memory on the CPU; absent evidence is never zero."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import plain

from .collectives import matching_measurements
from .requests import CollectiveRequest, collective_requests

if TYPE_CHECKING:
    from shadowspill.artifacts.program import Program

    from .info import CommunicationInfo


def collective_memory(
    requests: tuple[CollectiveRequest, ...],
    info: CommunicationInfo,
    setup: Mapping[str, Any],
    *,
    fallback_bytes: int | None = None,
    fallback_kind: str = "estimated",
) -> tuple[int, tuple[str, ...]]:
    sizes, reasons = [], []
    for request in requests:
        if len(request.ranks) == 1:
            sizes.append(0)
            continue
        rows = matching_measurements(
            info, {**request.to_dict(), "setup": setup, "topology": info.topology}
        )
        record = next(
            (
                row["memory"]
                for row in rows
                if row.get("memory", {}).get("workspace_bytes") is not None
                and row["memory"].get("memory_kind") != "unresolved"
            ),
            None,
        )
        size = fallback_bytes if record is None else record["workspace_bytes"]
        kind = (
            fallback_kind if record is None else record.get("memory_kind", "estimated")
        )
        if size is None:
            raise ValueError(
                "incomplete communication memory: workspace_bytes for "
                f"{request.operation} {request.ranks} ({request.tensor_bytes} bytes)"
            )
        sizes.append(size)
        if kind not in {"measured", "static"}:
            reasons.append(f"estimated collective workspace: {request.operation}")
    return max(sizes, default=0), tuple(dict.fromkeys(reasons))


def communication_overhead(
    program: Program, info: CommunicationInfo
) -> tuple[int, tuple[str, ...]]:
    requests = collective_requests(program, None)
    if not any(len(request.ranks) > 1 for request in requests):
        return 0, ()
    if info.fixed_device_bytes is None:
        raise ValueError("incomplete communication memory: fixed_device_bytes")
    reasons = []
    kind = info.metadata.get("memory_kind", "estimated")
    if kind not in {"measured", "static"}:
        reasons.append("estimated communicator overhead")
    if kind == "measured":
        scope = info.metadata["memory_scope"]
        expected = {
            "setup": plain(program.setup),
            "topology": plain(info.topology),
            "groups": [list(group) for group in sorted({r.ranks for r in requests})],
        }
        if any(plain(scope[key]) != value for key, value in expected.items()):
            raise ValueError("communicator memory belongs to different groups/setup")
        if plain(scope.get("backend_setup", {})) != plain(
            info.metadata.get("backend_setup", {})
        ):
            raise ValueError("communicator memory belongs to a different backend setup")
        if info.metadata.get("external_transient_kind") != "verified":
            reasons.append("external communication scratch is not certified")
    # Transfer callbacks allocate outside task scopes. Reserve their observed
    # scratch conservatively for every sharded object; never hide it as zero or
    # assume asynchronous backend frees have retired at the next submission.
    scratch = 0
    for shard in program.bindings.get("weight_shards", {}).values():
        size, unresolved = collective_memory(
            tuple(
                CollectiveRequest(
                    "all_gather",
                    tuple(group),
                    shard["padded_bytes"],
                    shard["dtype"],
                    inplace=True,
                )
                for group in shard["groups"]
            ),
            info,
            program.setup,
        )
        scratch += size
        reasons.extend(unresolved)
    return info.fixed_device_bytes + scratch, tuple(dict.fromkeys(reasons))
