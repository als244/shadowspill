"""Semantic task and byte/timing rows from a simulated Plan."""

from typing import Any

from shadowspill.artifacts.plan import Plan


def task_rows(plan: Plan) -> list[dict[str, Any]]:
    tasks = {item.task_id: item for item in plan.program.tasks}
    definitions = {item.profile_key: item for item in plan.program.task_definitions}
    rows = []
    for interval in plan.timeline.task_intervals:
        task = tasks[interval.task_id]
        definition = definitions[task.profile_id]
        rows.append(
            {
                "task_id": task.task_id,
                "name": definition.name or task.task_id,
                "binding": plan.program.bindings.get("task_names", {}).get(
                    task.task_id, ""
                ),
                "modules": ", ".join(definition.module_paths),
                "profile_key": definition.profile_key,
                "phase": task.phase,
                "start_seconds": interval.start_ns / 1e9,
                "duration_seconds": (interval.end_ns - interval.start_ns) / 1e9,
                "workspace_bytes": interval.workspace_bytes,
                "stall_seconds": interval.stall_ns / 1e9,
            }
        )
    return rows


def transfer_rows(plan: Plan) -> list[dict[str, Any]]:
    rows = []
    aliases = {item.alias_group_id: item for item in plan._costed_program.alias_groups}
    for item in plan.timeline.transfer_intervals:
        duration = (item.end_ns - item.start_ns) / 1e9
        transfer = aliases[item.alias_group_id].transfer
        phases = () if transfer is None else getattr(transfer, item.direction.value)
        collective_bytes = sum(phase.bytes for phase in phases if phase.kind == "timed")
        collective = any(phase.submission_ordered for phase in phases)
        rows.append(
            {
                "storage_id": item.alias_group_id,
                "source_names": ", ".join(
                    plan.program.sources.get(item.alias_group_id, ())
                ),
                "direction": item.direction.value,
                "kind": "copy_and_collective" if collective else "copy",
                "bytes": item.bytes + collective_bytes,
                "host_bytes": item.bytes,
                "collective_bytes": collective_bytes,
                "start_seconds": item.start_ns / 1e9,
                "duration_seconds": duration,
                # The native simulator retains the composite interval, not
                # individual leg timestamps. Do not report it as memcpy speed.
                "throughput_gb_per_second": None
                if collective
                else (item.bytes / duration / 1e9 if duration else 0.0),
                "stall_seconds": item.stall_ns / 1e9,
            }
        )
    return rows
