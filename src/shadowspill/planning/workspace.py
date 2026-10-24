"""Account for returned allocations that a bound task uses only transiently."""

from collections.abc import Mapping, Sequence
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.task.allocations import TaskAllocationEvent, TaskAllocationOperation
from shadowspill.task.layout import _allocation_live_set_peak


def transient_output_workspace(
    leaves: Sequence[int], manifest: Mapping[str, Any], metadata: Mapping[str, Any]
) -> int:
    if not leaves:
        return 0
    selected = set(leaves)
    if "allocation_trace" in metadata:
        trace = tuple(
            TaskAllocationEvent.from_dict(plain(event))
            for event in metadata["allocation_trace"]
        )
        ordinals = {
            event.allocation_ordinal
            for event in trace
            if event.operation is TaskAllocationOperation.ALLOCATE
            and selected.intersection(event.output_leaf_indices)
        }
        return _allocation_live_set_peak(trace, ordinals) - _allocation_live_set_peak(
            trace, set()
        )
    roots = {
        view["root_id"]
        for view in manifest["storage_contract"]["output_views"]
        if view["leaf_index"] in selected
    }
    # Without a physical trace, reserve the complete compiler output extents.
    # This is a conservative logical-simulation bound, not execution evidence.
    return sum(
        item["requested_bytes"]
        for item in manifest["root_allocations"]
        if item["root_id"] in roots
    )
