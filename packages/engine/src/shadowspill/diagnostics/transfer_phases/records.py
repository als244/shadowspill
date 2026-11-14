"""Index measured host-copy and collective events by their parent transfer."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

ParentRecords = dict[tuple[str, str | None, str | None], list[dict[str, Any]]]
Summaries = dict[tuple[str, str], list[dict[str, Any]]]

LABELS = {
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


def index_records(ledger: Mapping[str, Any]) -> tuple[ParentRecords, Summaries]:
    copy_ledger = ledger.get("host_copies") or {}
    by_parent: ParentRecords = defaultdict(list)
    summaries: Summaries = defaultdict(list)
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
                "label": LABELS.get(operation, operation),
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

    return by_parent, summaries
