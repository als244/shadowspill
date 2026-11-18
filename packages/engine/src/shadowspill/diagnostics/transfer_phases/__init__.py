"""Separate measured host copies, collectives, and unresolved transfer time.

A composite lane interval is not a memcpy measurement. The lane's own marks
bracket the copy and a collective's events bracket the collective; missing
times remain unresolved.
"""

from collections.abc import Mapping
from typing import Any

from .records import index_records
from .summary import summarize
from .timeline import copy_only, match_parts, regions


def _copies_whole_payload(
    geometry: Mapping[str, Any], direction: str, host_bytes: int
) -> bool:
    """Whether this direction's lane interval is one copy of the entire payload.

    An alias without a recipe is a plain copy. Planning also prices a device's
    host latency as a timed phase ahead of that copy; the lane still performs
    one untransformed copy, so its interval remains the copy's measurement.
    Any other timed phase, a split copy, or a different size is a transformation.
    """

    recipe = geometry.get("transfer")
    if recipe is None:
        return True
    phases = recipe.get(direction, ())
    copies = [phase for phase in phases if phase["kind"] == "copy"]
    return (
        len(copies) == 1
        and copies[0]["bytes"] == host_bytes
        and all(
            phase["kind"] == "copy" or phase.get("timing_class") == "host_latency"
            for phase in phases
        )
    )


def transfer_phase_data(
    program: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    """Return timeline regions and payload/rate summaries on the device clock.

    Parent bytes are context, never added to individual operation totals. A
    copy-only lane can supply its copy interval; a transformed transfer needs
    explicit copy events. Missing events remain a gray, unpriced region.
    """

    for direction in ("fetch", "evict"):
        for transfer in diagnostics.get("transfers", {}).get(direction, {}).values():
            missing = {"alias_group_id", "transfer_id", "bytes"} - transfer.keys()
            if missing:
                raise ValueError(f"{direction} transfer is missing {sorted(missing)}")
    aliases = {a["alias_group_id"]: a for a in program.get("alias_groups", ())}
    ledger = diagnostics.get("collectives") or {}
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
            if not parts and _copies_whole_payload(
                geometry, direction, transfer["bytes"]
            ):
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
    # False when the capture holds no record of the copies inside transformed
    # transfers at all, as against holding none because there were none.
    result["copies_recorded"] = ledger.get("host_copies") is not None
    result["summary"] = summarize(summaries)
    return result
