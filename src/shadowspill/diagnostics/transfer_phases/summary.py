"""Compute logical payload throughput and separate busy-union from summed time."""

from typing import Any

from .records import LABELS, Summaries


def _busy_ns(records: list[dict[str, Any]]) -> int:
    end = total = 0
    for start, stop in sorted((r["start_ns"], r["end_ns"]) for r in records):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(summaries: Summaries) -> list[dict[str, Any]]:
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
                "label": LABELS.get(operation, operation),
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
    return summary
