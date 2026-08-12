"""Extract the Phase-1 execution decomposition from an NSYS SQLite export."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

_TASK = re.compile(r"^shadowspill\.task\.(forward|backward|optimizer)\.(task_[0-9]+)$")
_SEGMENTS = (
    "before_task",
    "storage_rebind",
    "compiled_call",
    "after_task",
)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _ranges(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(connection, "NVTX_EVENTS"):
        raise ValueError("NSYS export has no NVTX_EVENTS table")
    rows = connection.execute(
        """
        SELECT events.start, events.end, events.globalTid,
               COALESCE(events.text, names.value)
        FROM NVTX_EVENTS AS events
        LEFT JOIN StringIds AS names ON names.id = events.textId
        WHERE events.end IS NOT NULL
          AND COALESCE(events.text, names.value) LIKE 'shadowspill.%'
        ORDER BY events.start, events.end
        """
    ).fetchall()
    return [
        {
            "start_ns": int(start),
            "end_ns": int(end),
            "global_tid": int(global_tid),
            "name": str(name),
        }
        for start, end, global_tid, name in rows
    ]


def _correlated_gpu_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    start_ns: int,
    end_ns: int,
    global_tid: int,
) -> list[tuple[int, int, int, str]]:
    if not _table_exists(connection, table) or not _table_exists(
        connection, "CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        return []
    if table == "CUPTI_ACTIVITY_KIND_KERNEL":
        name_join = "JOIN StringIds AS names ON names.id = gpu.shortName"
        name_value = "names.value"
    else:
        name_join = "JOIN ENUM_CUDA_MEMCPY_OPER AS names ON names.id = gpu.copyKind"
        name_value = "names.label"
    query = f"""
        SELECT gpu.start, gpu.end, gpu.streamId, {name_value}
        FROM {table} AS gpu
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
          ON runtime.correlationId = gpu.correlationId
        {name_join}
        WHERE runtime.globalTid = ?
          AND runtime.start >= ?
          AND runtime.start <= ?
        ORDER BY gpu.start, gpu.end
    """
    return [
        (int(start), int(end), int(stream), str(name))
        for start, end, stream, name in connection.execute(
            query, (global_tid, start_ns, end_ns)
        )
    ]


def _merged_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += end - start
        start, end = next_start, next_end
    return total + end - start


def _idle_duration(intervals: list[tuple[int, int]]) -> int:
    if len(intervals) < 2:
        return 0
    ordered = sorted(intervals)
    idle = 0
    end = ordered[0][1]
    for start, next_end in ordered[1:]:
        idle += max(0, start - end)
        end = max(end, next_end)
    return idle


def _api_counts(connection: sqlite3.Connection) -> dict[str, int]:
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return {}
    rows = connection.execute(
        """
        SELECT names.value, count(*)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS names ON names.id = runtime.nameId
        WHERE names.value LIKE '%Event%'
           OR names.value LIKE '%StreamWaitEvent%'
           OR names.value LIKE '%Synchronize%'
        GROUP BY names.value
        ORDER BY names.value
        """
    ).fetchall()
    return {str(name): int(count) for name, count in rows}


def extract_trace(path: Path) -> dict[str, object]:
    """Return a deterministic execution summary for one NSYS SQLite export."""

    with sqlite3.connect(path) as connection:
        ranges = _ranges(connection)
        segment_by_task: dict[tuple[str, str], dict[str, Any]] = {}
        for item in ranges:
            name = str(item["name"])
            for segment in _SEGMENTS:
                prefix = f"shadowspill.{segment}."
                if name.startswith(prefix):
                    segment_by_task[(name.removeprefix(prefix), segment)] = item

        tasks: list[dict[str, object]] = []
        all_kernels: set[tuple[int, int, int, str]] = set()
        phase_kernel_ns: dict[str, int] = defaultdict(int)
        for item in ranges:
            match = _TASK.fullmatch(str(item["name"]))
            if match is None:
                continue
            phase, task_id = match.groups()
            compiled = segment_by_task.get((task_id, "compiled_call"))
            kernels = (
                []
                if compiled is None
                else _correlated_gpu_rows(
                    connection,
                    table="CUPTI_ACTIVITY_KIND_KERNEL",
                    start_ns=int(compiled["start_ns"]),
                    end_ns=int(compiled["end_ns"]),
                    global_tid=int(compiled["global_tid"]),
                )
            )
            all_kernels.update(kernels)
            kernel_ns = sum(end - start for start, end, _stream, _name in kernels)
            phase_kernel_ns[phase] += kernel_ns
            segments = {
                segment: (
                    int(value["end_ns"]) - int(value["start_ns"])
                    if (value := segment_by_task.get((task_id, segment))) is not None
                    else 0
                )
                for segment in _SEGMENTS
            }
            tasks.append(
                {
                    "task_id": task_id,
                    "phase": phase,
                    "host_start_ns": int(item["start_ns"]),
                    "host_end_ns": int(item["end_ns"]),
                    "host_duration_ns": int(item["end_ns"]) - int(item["start_ns"]),
                    "host_segments_ns": segments,
                    "kernel_count": len(kernels),
                    "kernel_sum_ns": kernel_ns,
                    "gpu_start_ns": min((row[0] for row in kernels), default=None),
                    "gpu_end_ns": max((row[1] for row in kernels), default=None),
                    "streams": sorted({row[2] for row in kernels}),
                }
            )

        tasks.sort(
            key=lambda value: (
                int(value["host_start_ns"]),
                str(value["task_id"]),
            )
        )
        for index, task in enumerate(tasks):
            if index == 0:
                task["host_gap_from_prior_task_ns"] = 0
            else:
                task["host_gap_from_prior_task_ns"] = max(
                    0,
                    int(task["host_start_ns"]) - int(tasks[index - 1]["host_end_ns"]),
                )

        kernels_by_stream: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for start, end, stream, _name in all_kernels:
            kernels_by_stream[stream].append((start, end))
        compute_stream = (
            max(
                kernels_by_stream,
                key=lambda stream: _merged_duration(kernels_by_stream[stream]),
            )
            if kernels_by_stream
            else None
        )
        compute_intervals = (
            kernels_by_stream[compute_stream] if compute_stream is not None else []
        )
        compute_start = min((start for start, _end in compute_intervals), default=None)
        compute_end = max((end for _start, end in compute_intervals), default=None)

        transfer_rows: list[tuple[int, int, int, str]] = []
        transfer_dispatch_ns: dict[str, int] = defaultdict(int)
        for item in ranges:
            name = str(item["name"])
            if name not in {
                "shadowspill.runtime.transfer.h2d",
                "shadowspill.runtime.transfer.d2h",
            }:
                continue
            direction = name.rsplit(".", 1)[-1]
            transfer_dispatch_ns[direction] += int(item["end_ns"]) - int(
                item["start_ns"]
            )
            transfer_rows.extend(
                _correlated_gpu_rows(
                    connection,
                    table="CUPTI_ACTIVITY_KIND_MEMCPY",
                    start_ns=int(item["start_ns"]),
                    end_ns=int(item["end_ns"]),
                    global_tid=int(item["global_tid"]),
                )
            )
        transfer_rows = sorted(set(transfer_rows))
        transfer_summary: dict[str, dict[str, int]] = {}
        for label in sorted({row[3] for row in transfer_rows}):
            selected = [row for row in transfer_rows if row[3] == label]
            overlap_ns = sum(
                max(0, min(end, kernel_end) - max(start, kernel_start))
                for start, end, _stream, _name in selected
                for kernel_start, kernel_end in compute_intervals
            )
            transfer_summary[label] = {
                "count": len(selected),
                "duration_ns": sum(end - start for start, end, _stream, _ in selected),
                "compute_overlap_ns": overlap_ns,
            }

        optimizer = [task for task in tasks if task["phase"] == "optimizer"]
        optimizer_starts = [
            int(task["gpu_start_ns"])
            for task in optimizer
            if task["gpu_start_ns"] is not None
        ]
        optimizer_ends = [
            int(task["gpu_end_ns"])
            for task in optimizer
            if task["gpu_end_ns"] is not None
        ]
        return {
            "schema": "shadowspill.execution_trace/v1",
            "source": str(path.resolve()),
            "task_count": len(tasks),
            "optimizer_task_count": len(optimizer),
            "compute_stream_id": compute_stream,
            "compute_start_ns": compute_start,
            "compute_end_ns": compute_end,
            "compute_span_ns": (
                compute_end - compute_start
                if compute_start is not None and compute_end is not None
                else 0
            ),
            "compute_kernel_union_ns": _merged_duration(compute_intervals),
            "compute_idle_ns": _idle_duration(compute_intervals),
            "optimizer_start_ns": min(optimizer_starts, default=None),
            "optimizer_end_ns": max(optimizer_ends, default=None),
            "optimizer_span_ns": (
                max(optimizer_ends) - min(optimizer_starts)
                if optimizer_starts and optimizer_ends
                else 0
            ),
            "phase_kernel_ns": dict(sorted(phase_kernel_ns.items())),
            "transfer_dispatch_ns": dict(sorted(transfer_dispatch_ns.items())),
            "transfers": transfer_summary,
            "cuda_api_counts": _api_counts(connection),
            "tasks": tasks,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ShadowSpill task, kernel, transfer, and idle timings."
    )
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = extract_trace(arguments.sqlite.resolve())
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
