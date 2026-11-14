"""Match events to one lane interval and retain its unexplained gaps."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from .records import LABELS


def match_parts(
    records: Sequence[Mapping[str, Any]],
    matched: set[tuple[str, int]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    parts = []
    for record in records:
        if (
            record["key"] not in matched
            and start <= record["start_ns"] <= record["end_ns"] <= end
        ):
            matched.add(record["key"])
            parts.append({k: v for k, v in record.items() if k != "key"})
    return parts


def copy_only(context: Mapping[str, Any]) -> dict[str, Any]:
    direction = context["source"]
    operation = "fetch_copy" if direction == "fetch" else "evict_copy"
    return {
        "kind": "copy",
        "operation": operation,
        "label": LABELS[operation],
        "start_ns": context["parent_start_ns"],
        "end_ns": context["parent_end_ns"],
        "bytes": context["host_bytes"],
        "input_bytes": None,
        "output_bytes": None,
        "timing": "copy-only lane events",
        "source": direction,
        "record_id": context["transfer_id"],
    }


def regions(
    context: Mapping[str, Any], parts: Sequence[Mapping[str, Any]], copy_complete: bool
) -> Iterator[dict[str, Any]]:
    cursor, end = context["parent_start_ns"], context["parent_end_ns"]
    for part in sorted(parts, key=lambda p: p["start_ns"]):
        if cursor < part["start_ns"]:
            yield _gap(context, cursor, part["start_ns"], copy_complete)
        yield {**context, **part}
        cursor = max(cursor, part["end_ns"])
    if cursor < end:
        yield _gap(context, cursor, end, copy_complete)


def _gap(
    context: Mapping[str, Any], start: int, end: int, copy_complete: bool
) -> dict[str, Any]:
    operation = "overhead" if copy_complete else "unresolved"
    return {
        **context,
        "kind": "unresolved",
        "operation": operation,
        "label": LABELS[operation],
        "start_ns": start,
        "end_ns": end,
        "bytes": None,
        "timing": "unresolved lane interval",
    }
