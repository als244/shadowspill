"""Validate plain communication memory records without importing GPU code."""

from collections.abc import Mapping, Sequence
from typing import Any

from shadowspill.artifacts._values import nonnegative_int, plain
from shadowspill.task.allocations import TaskAllocationContract, TaskAllocationEvent
from shadowspill.task.layout import _allocation_live_set_peak


def validate_task_memory(memory: Mapping[str, Any], ranks: Sequence[int]) -> None:
    kind = memory.get("memory_kind", "estimated")
    if kind not in {"measured", "estimated", "static", "unresolved"}:
        raise ValueError("invalid collective memory kind")
    workspace = memory.get("workspace_bytes")
    if workspace is not None:
        nonnegative_int(workspace, "collective workspace_bytes")
    if kind == "unresolved":
        return
    if workspace is None:
        raise ValueError("collective memory record requires workspace_bytes")
    if kind != "measured":
        return
    rows = memory.get("per_rank", ())
    if sorted(row["logical_rank"] for row in rows) != sorted(ranks):
        raise ValueError("collective memory needs one record per participant")
    for row in rows:
        if row.get("memory_kind") != "measured":
            raise ValueError("collective memory contains an unmeasured participant")
        trace = tuple(
            TaskAllocationEvent.from_dict(plain(event))
            for event in row["allocation_trace"]
        )
        contract = TaskAllocationContract.from_dict(plain(row["allocation_contract"]))
        if TaskAllocationContract.capture(trace) != contract:
            raise ValueError("collective allocation trace differs from its contract")
        if _allocation_live_set_peak(trace, set()) != row["workspace_bytes"]:
            raise ValueError("collective workspace differs from its allocation trace")
    if workspace != max(row["workspace_bytes"] for row in rows):
        raise ValueError("collective workspace must cover every participant")


def validate_fixed_memory(size: int | None, metadata: Mapping[str, Any]) -> None:
    if size is not None:
        nonnegative_int(size, "communication fixed_device_bytes")
    if metadata.get("memory_kind", "estimated") != "measured":
        return
    scope = metadata.get("memory_scope", {})
    if (
        size is None
        or not scope.get("setup")
        or scope.get("backend") != "nccl"
        or not scope.get("topology")
        or not scope.get("groups")
    ):
        raise ValueError("measured communicator memory requires complete scope")
    ranks = [rank for group in scope["topology"].values() for rank in group]
    if len(ranks) != len(set(ranks)):
        raise ValueError("communication memory topology contains duplicate ranks")
    for group in scope["groups"]:
        if not group or len(group) != len(set(group)) or not set(group) <= set(ranks):
            raise ValueError("communication memory group differs from topology")
    rows = metadata.get("memory_per_rank", ())
    if sorted(row["logical_rank"] for row in rows) != sorted(ranks):
        raise ValueError("communicator memory needs one record per logical rank")
    for row in rows:
        for field in ("native_retained_bytes", "external_retained_bytes"):
            nonnegative_int(row[field], field)
        if row["native_retained_bytes"] + row["external_retained_bytes"] > size:
            raise ValueError("communicator memory exceeds fixed_device_bytes")
