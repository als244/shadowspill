"""Separate measured host copies, collectives, and unresolved transfer time.

A composite lane interval is not a memcpy measurement. Historical captures
can locate NCCL work, but their remaining time stays unresolved. New captures
also bracket the host copy inside transformed transfers with device events.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

_LABELS = {
    "fetch_copy": "host to device copy",
    "evict_copy": "device to host copy",
    "all_gather": "all-gather",
    "reduce_scatter": "reduce-scatter",
    "all_reduce": "all-reduce",
    "unresolved": "copy / dispatch / wait (unsplit)",
    "overhead": "dispatch / wait / other work",
}


def _times(record: Mapping[str, Any]) -> tuple[int, int] | None:
    start, end = record.get("started_at_ns"), record.get("finished_at_ns")
    if start is None or end is None or end < start:
        return None
    return int(start), int(end)


def _busy_ns(records: list[dict[str, Any]]) -> int:
    end = total = 0
    for start, stop in sorted((r["start_ns"], r["end_ns"]) for r in records):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def transfer_phase_data(
    program: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return timeline regions and payload/rate summaries on the device clock.

    Parent bytes are context, never added to individual operation totals. A
    copy-only lane can supply its copy interval; a transformed transfer needs
    explicit copy events. Missing events remain a gray, unpriced region.
    """

    # Older/minimal diagnostics may retain only sequence and lane clocks.
    # Keep the original composite view rather than inventing ownership/bytes.
    for direction in ("fetch", "evict"):
        for transfer in diagnostics.get("transfers", {}).get(direction, {}).values():
            if not {"alias_group_id", "transfer_id", "bytes"} <= transfer.keys():
                return None
    aliases = {a["alias_group_id"]: a for a in program.get("alias_groups", ())}
    ledger = diagnostics.get("collectives") or {}
    copy_ledger = ledger.get("host_copies") or {}
    by_parent: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    summaries: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    result: dict[str, Any] = {"fetch": [], "evict": [], "unmatched_records": 0}
    matched: set[tuple[str, int]] = set()
    for kind, records in (
        ("collective", ledger.get("records", ())),
        ("copy", copy_ledger.get("records", ())),
    ):
        for index, record in enumerate(records):
            times = _times(record)
            if times is None:
                continue
            source = record.get("source", "unknown")
            operation = (
                record.get("operation")
                if kind == "collective"
                else ("fetch_copy" if source == "fetch" else "evict_copy")
            )
            item = {
                "kind": kind,
                "operation": operation,
                "label": _LABELS.get(operation, operation),
                "start_ns": times[0],
                "end_ns": times[1],
                "bytes": record.get("bytes"),
                "input_bytes": record.get("input_bytes"),
                "output_bytes": record.get("output_bytes"),
                "source": source,
                "timing": "collective events"
                if kind == "collective"
                else "copy device events",
                "record_id": record.get("collective_id", record.get("copy_id")),
                "group": record.get("group"),
                "group_size": record.get("group_size"),
                "group_sequence": record.get("group_sequence"),
                "key": (kind, index),
            }
            if kind == "collective":
                summaries[(source, operation)].append(item)
            if source in ("fetch", "evict"):
                by_parent[
                    (
                        source,
                        record.get("alias_group_id"),
                        record.get("execution_task_id"),
                    )
                ].append(item)

    unresolved_bytes = {"fetch": 0, "evict": 0}
    unresolved_transfers = {"fetch": 0, "evict": 0}
    for direction in ("fetch", "evict"):
        for transfer in diagnostics.get("transfers", {}).get(direction, {}).values():
            lane = transfer.get("lane", {})
            start = lane.get("lane_started_at_seconds")
            end = lane.get("lane_finished_at_seconds")
            if start is None or end is None:
                continue
            start, end = round(start * 1e9), round(end * 1e9)
            if end <= start:
                continue
            alias = transfer["alias_group_id"]
            geometry = aliases.get(alias, {})
            context = {
                "transfer_id": transfer["transfer_id"],
                "alias_group_id": alias,
                "source": direction,
                "parent_start_ns": start,
                "parent_end_ns": end,
                "host_bytes": transfer["bytes"],
                "execution_bytes": geometry.get("size_bytes"),
            }
            key = (direction, alias, transfer.get("triggered_by"))
            parts = []
            for record in by_parent.get(key, ()):
                if record["key"] in matched:
                    continue
                if start <= record["start_ns"] <= record["end_ns"] <= end:
                    matched.add(record["key"])
                    parts.append({k: v for k, v in record.items() if k != "key"})
            copies = [part for part in parts if part["kind"] == "copy"]
            recipe = geometry.get("transfer")
            # Any explicit transform needs its own events, including a
            # copy-only shard slice: it may also enqueue memset/preparation.
            if not copies and recipe is None and not parts:
                operation = "fetch_copy" if direction == "fetch" else "evict_copy"
                copies = [
                    {
                        "kind": "copy",
                        "operation": operation,
                        "label": _LABELS[operation],
                        "start_ns": start,
                        "end_ns": end,
                        "bytes": transfer["bytes"],
                        "input_bytes": None,
                        "output_bytes": None,
                        "timing": "copy-only lane events",
                        "source": direction,
                        "record_id": transfer["transfer_id"],
                    }
                ]
                parts.extend(copies)
            copy_complete = sum(copy["bytes"] for copy in copies) == transfer["bytes"]
            if not copy_complete:
                unresolved_bytes[direction] += transfer["bytes"]
                unresolved_transfers[direction] += 1
            for copy in copies:
                summaries[(direction, copy["operation"])].append(copy)
            cursor = start
            for part in sorted(parts, key=lambda p: p["start_ns"]):
                if cursor < part["start_ns"]:
                    operation = "overhead" if copy_complete else "unresolved"
                    result[direction].append(
                        {
                            **context,
                            "kind": "unresolved",
                            "operation": operation,
                            "label": _LABELS[operation],
                            "start_ns": cursor,
                            "end_ns": part["start_ns"],
                            "bytes": None,
                            "timing": "unresolved lane interval",
                        }
                    )
                result[direction].append({**context, **part})
                cursor = max(cursor, part["end_ns"])
            if cursor < end:
                operation = "overhead" if copy_complete else "unresolved"
                result[direction].append(
                    {
                        **context,
                        "kind": "unresolved",
                        "operation": operation,
                        "label": _LABELS[operation],
                        "start_ns": cursor,
                        "end_ns": end,
                        "bytes": None,
                        "timing": "unresolved lane interval",
                    }
                )
    result["unmatched_records"] = sum(
        r["key"] not in matched for records in by_parent.values() for r in records
    )
    result["unresolved_host_bytes"] = unresolved_bytes
    result["unresolved_copy_transfers"] = unresolved_transfers
    result["copy_capture"] = (
        {
            k: copy_ledger.get(k)
            for k in ("capacity", "complete", "dropped", "errors", "missing_timings")
        }
        if ledger.get("host_copies") is not None
        else None
    )
    summary = []
    for (source, operation), records in sorted(summaries.items()):
        busy = _busy_ns(records)
        copied = sum(r["bytes"] or 0 for r in records)
        # Logical collective payload: assembled all-gather output, full
        # reduce-scatter input, or one all-reduce payload. This is algorithmic
        # throughput, not a sum of input/output bytes or physical link traffic.
        payload = (
            copied
            if records[0]["kind"] == "copy"
            else sum(
                max(r["input_bytes"] or 0, r["output_bytes"] or 0) for r in records
            )
        )
        summary.append(
            {
                "source": source,
                "operation": operation,
                "label": _LABELS.get(operation, operation),
                "count": len(records),
                "busy_ns": busy,
                "duration_sum_ns": sum(r["end_ns"] - r["start_ns"] for r in records),
                "copy_bytes": copied if records[0]["kind"] == "copy" else None,
                "input_bytes": sum(r["input_bytes"] or 0 for r in records)
                if records[0]["kind"] == "collective"
                else None,
                "output_bytes": sum(r["output_bytes"] or 0 for r in records)
                if records[0]["kind"] == "collective"
                else None,
                "throughput_bytes": payload,
                "throughput_gbps": payload / busy if busy else None,
                "host_copy_gbps": copied / busy if copied and busy else None,
            }
        )
    result["summary"] = summary
    return result
