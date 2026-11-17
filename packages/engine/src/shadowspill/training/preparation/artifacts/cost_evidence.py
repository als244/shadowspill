"""Allow labeled timing edits while retaining original code and memory facts."""

from collections.abc import Mapping
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.profiles import Profiles
from shadowspill.communication.info import CommunicationInfo


def _edits(
    original: Mapping[str, Any],
    changed: Mapping[str, Any],
    fields: set[str],
) -> dict[str, Any]:
    if {k: v for k, v in original.items() if k not in fields} != {
        k: v for k, v in changed.items() if k not in fields
    }:
        raise ValueError("local planning changed physical evidence or its provenance")
    return {
        key: {"measured": original.get(key), "estimated": changed.get(key)}
        for key in fields
        if original.get(key) != changed.get(key)
    }


def profile_edits(original: Profiles, changed: Profiles) -> dict[str, Any]:
    before, after = original.to_dict(), changed.to_dict()
    tasks, values = before.pop("tasks"), after.pop("tasks")
    if before != after or tasks.keys() != values.keys():
        raise ValueError("local planning changed measured setup, memory or task forms")
    result = {}
    for key, source in tasks.items():
        value = values[key]
        edits = _edits(
            source,
            value,
            {
                "duration_seconds",
                "resource_phases",
                "overlap_costs",
                "metadata",
            },
        )
        # Resource names, message geometry and progress classes remain measured.
        for field, allowed in (
            ("resource_phases", {"duration_ns"}),
            ("overlap_costs", {"task_rate", "transfer_rate"}),
        ):
            left, right = source[field] or [], value[field] or []
            if len(left) != len(right):
                raise ValueError("local planning changed measured resource geometry")
            for a, b in zip(left, right, strict=True):
                _edits(a, b, allowed)
        _edits(source["metadata"], value["metadata"], {"duration_kind"})
        edits.pop("metadata", None)
        if edits:
            if value["metadata"].get("duration_kind") != "estimated":
                raise ValueError("edited task timing must be labeled estimated")
            result[key] = edits
        elif value["metadata"] != source["metadata"]:
            raise ValueError(
                "unchanged timing must preserve its measured classification"
            )
    return result


def communication_edits(
    original: CommunicationInfo,
    changed: CommunicationInfo,
) -> dict[str, Any]:
    before, after = original.to_dict(), changed.to_dict()
    fields = {
        name
        for name in before
        if name.endswith(("_bytes_per_second", "_latency_seconds"))
    }
    edits = _edits(before, after, fields | {"metadata", "measurements"})
    _edits(before["metadata"], after["metadata"], {"timing_kind"})
    left, right = before["measurements"], after["measurements"]
    if len(left) != len(right):
        raise ValueError("local planning changed collective measurement geometry")
    samples = {}
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        delta = _edits(a, b, {"duration_seconds", "zero_duration_seconds"})
        if delta:
            samples[str(index)] = delta
    edits.pop("metadata", None)
    edits.pop("measurements", None)
    if samples:
        edits["measurements"] = samples
    if edits:
        if changed.metadata.get("timing_kind") != "estimated":
            raise ValueError("edited communication timing must be labeled estimated")
    elif plain(original.metadata) != plain(changed.metadata):
        raise ValueError("unchanged timing must preserve its measured classification")
    return edits
