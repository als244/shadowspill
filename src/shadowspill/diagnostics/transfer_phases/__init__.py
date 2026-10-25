"""Separate measured host copies, collectives, and unresolved transfer time.

A composite lane interval is not a memcpy measurement. Historical captures
can locate NCCL work, but their remaining time stays unresolved. New captures
also bracket the host copy inside transformed transfers with device events.
"""

from collections.abc import Mapping
from typing import Any

from .records import index_records
from .summary import summarize
from .timeline import copy_only, match_parts, regions


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
    by_parent, summaries = index_records(ledger)
    result: dict[str, Any] = {"fetch": [], "evict": [], "unmatched_records": 0}
    matched: set[tuple[str, int]] = set()
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
            parts = match_parts(by_parent.get(key, ()), matched, start, end)
            copies = [part for part in parts if part["kind"] == "copy"]
            # Transformed transfers need their own copy events: even a shard
            # slice can include memset/preparation before the physical copy.
            if not copies and geometry.get("transfer") is None and not parts:
                copies = [copy_only(context)]
                parts.extend(copies)
            copy_complete = sum(copy["bytes"] for copy in copies) == transfer["bytes"]
            if not copy_complete:
                unresolved_bytes[direction] += transfer["bytes"]
                unresolved_transfers[direction] += 1
            for copy in copies:
                summaries[(direction, copy["operation"])].append(copy)
            result[direction].extend(regions(context, parts, copy_complete))
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
    result["summary"] = summarize(summaries)
    return result
