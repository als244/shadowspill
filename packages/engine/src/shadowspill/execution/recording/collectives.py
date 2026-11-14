"""NCCL event capture for a single explicitly traced invocation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from shadowspill.diagnostics.collectives import (
    CollectiveRecord,
    CollectiveTrace,
    HostCopyRecord,
    HostCopyTrace,
)

if TYPE_CHECKING:
    from shadowspill.diagnostics.timing import ArmedExecutionTiming
    from shadowspill.runtime.plan.bridge import RuntimeBridge


_OPERATIONS = {
    "_allgather_base_": "all_gather",
    "allgather_": "all_gather",
    "allgather_into_tensor_coalesced_": "all_gather",
    "allgather_coalesced_": "all_gather",
    "_reduce_scatter_base_": "reduce_scatter",
    "reduce_scatter_": "reduce_scatter",
    "reduce_scatter_tensor_coalesced_": "reduce_scatter",
    "allreduce_": "all_reduce",
    "allreduce_coalesced_": "all_reduce",
    "broadcast_": "broadcast",
}


def decode_collectives(
    payload: Mapping[str, Any],
    *,
    groups: Mapping[str, tuple[int, ...]],
    tasks: Mapping[int, tuple[str, str]],
    aliases: Mapping[int, str],
) -> CollectiveTrace:
    records = []
    for index, item in enumerate(payload.get("records", ())):
        task, execution = tasks.get(item["task_id"], (None, None))
        records.append(
            CollectiveRecord(
                collective_id=f"collective_{index:06d}",
                operation=_OPERATIONS.get(item["operation"], item["operation"]),
                group=item["group"],
                group_ranks=groups.get(item["group"], ()),
                group_rank=item["group_rank"],
                group_size=item["group_size"],
                group_sequence=item["group_sequence"],
                input_bytes=item["input_bytes"],
                output_bytes=item["output_bytes"],
                source=item["source"],
                task_id=task,
                execution_task_id=execution,
                object_id=item["object_id"],
                alias_group_id=aliases.get(item["object_id"]),
                action_ordinal=item["action_ordinal"],
                issued_at_ns=item["issued_at_ns"],
                started_at_ns=item["started_at_ns"],
                finished_at_ns=item["finished_at_ns"],
            )
        )
    copy_payload = payload.get("host_copies")
    copies = None
    if copy_payload is not None:
        copy_records = []
        for index, item in enumerate(copy_payload.get("records", ())):
            task, execution = tasks.get(item["task_id"], (None, None))
            copy_records.append(
                HostCopyRecord(
                    copy_id=f"host_copy_{index:06d}",
                    source=item["source"],
                    bytes=item["bytes"],
                    task_id=task,
                    execution_task_id=execution,
                    object_id=item["object_id"],
                    alias_group_id=aliases.get(item["object_id"]),
                    action_ordinal=item["action_ordinal"],
                    issued_at_ns=item["issued_at_ns"],
                    started_at_ns=item["started_at_ns"],
                    finished_at_ns=item["finished_at_ns"],
                )
            )
        copies = HostCopyTrace(
            capacity=copy_payload.get("capacity", 0),
            dropped=copy_payload.get("dropped", 0),
            errors=copy_payload.get("errors", 0),
            missing_timings=copy_payload.get("missing_timings", 0),
            records=tuple(copy_records),
        )
    return CollectiveTrace(
        enabled=True,
        capacity=payload.get("capacity", 0),
        dropped=payload.get("dropped", 0),
        errors=payload.get("errors", 0),
        missing_timings=payload.get("missing_timings", 0),
        records=tuple(records),
        host_copies=copies,
    )


def begin_collective_trace(
    bridge: RuntimeBridge, timing: ArmedExecutionTiming, *, capacity: int
) -> Callable[[], CollectiveTrace] | None:
    if not dist.is_initialized():
        return None
    groups = {
        str(group.group_name): tuple(dist.get_process_group_ranks(group))
        for group in dist.distributed_c10d._world.pg_map
    }
    tasks = {
        int(task_id.removeprefix("task_")): (
            task_id,
            f"execution_{task.execution_ordinal:06d}",
        )
        for task_id, task in timing.tasks.items()
    }
    aliases = {
        bridge.objects.runtime_object_id(alias): alias
        for alias in timing.alias_accesses
        if bridge.objects.requires_storage(alias)
    }
    torch.ops.shadowspill._collective_trace_begin(capacity)

    def finish() -> CollectiveTrace:
        return decode_collectives(
            torch.ops.shadowspill._collective_trace_end(),
            groups=groups,
            tasks=tasks,
            aliases=aliases,
        )

    return finish
