"""Serialization for simulator results and timelines."""

from __future__ import annotations

from shadowspill.ir import ResourceKind
from shadowspill.simulator import (
    DeviceMemoryPeak,
    MemorySnapshot,
    SimulationResult,
    TaskInterval,
    TransferDirection,
    TransferInterval,
)

from .common import _integer, _integer_pairs, _list, _mapping, _string, _string_tuple


def _simulation_result_from_value(value: object, path: str) -> SimulationResult:
    data = _mapping(value, path)
    tasks = _list(data.get("task_intervals"), f"{path}.task_intervals")
    transfers = _list(data.get("transfer_intervals"), f"{path}.transfer_intervals")
    peaks = _list(data.get("device_peaks"), f"{path}.device_peaks")
    timeline = _list(data.get("memory_timeline"), f"{path}.memory_timeline")
    return SimulationResult(
        makespan_ns=_integer(data.get("makespan_ns"), f"{path}.makespan_ns"),
        task_intervals=tuple(
            TaskInterval(
                task_id=_string(
                    item.get("task_id"), f"{path}.task_intervals[{index}].task_id"
                ),
                device_id=_string(
                    item.get("device_id"), f"{path}.task_intervals[{index}].device_id"
                ),
                resource_kind=ResourceKind(
                    _string(
                        item.get("resource_kind"),
                        f"{path}.task_intervals[{index}].resource_kind",
                    )
                ),
                resource_lane=_integer(
                    item.get("resource_lane"),
                    f"{path}.task_intervals[{index}].resource_lane",
                ),
                ready_ns=_integer(
                    item.get("ready_ns"), f"{path}.task_intervals[{index}].ready_ns"
                ),
                start_ns=_integer(
                    item.get("start_ns"), f"{path}.task_intervals[{index}].start_ns"
                ),
                end_ns=_integer(
                    item.get("end_ns"), f"{path}.task_intervals[{index}].end_ns"
                ),
                workspace_bytes=_integer(
                    item.get("workspace_bytes"),
                    f"{path}.task_intervals[{index}].workspace_bytes",
                ),
                stall_reasons=_string_tuple(
                    item.get("stall_reasons"),
                    f"{path}.task_intervals[{index}].stall_reasons",
                ),
            )
            for index, raw in enumerate(tasks)
            for item in (_mapping(raw, f"{path}.task_intervals[{index}]"),)
        ),
        transfer_intervals=tuple(
            TransferInterval(
                alias_group_id=_string(
                    item.get("alias_group_id"),
                    f"{path}.transfer_intervals[{index}].alias_group_id",
                ),
                trigger_task_id=_string(
                    item.get("trigger_task_id"),
                    f"{path}.transfer_intervals[{index}].trigger_task_id",
                ),
                device_id=_string(
                    item.get("device_id"),
                    f"{path}.transfer_intervals[{index}].device_id",
                ),
                direction=TransferDirection(
                    _string(
                        item.get("direction"),
                        f"{path}.transfer_intervals[{index}].direction",
                    )
                ),
                sequence=_integer(
                    item.get("sequence"),
                    f"{path}.transfer_intervals[{index}].sequence",
                ),
                ready_ns=_integer(
                    item.get("ready_ns"),
                    f"{path}.transfer_intervals[{index}].ready_ns",
                ),
                start_ns=_integer(
                    item.get("start_ns"),
                    f"{path}.transfer_intervals[{index}].start_ns",
                ),
                end_ns=_integer(
                    item.get("end_ns"),
                    f"{path}.transfer_intervals[{index}].end_ns",
                ),
                bytes=_integer(
                    item.get("bytes"), f"{path}.transfer_intervals[{index}].bytes"
                ),
                stall_reasons=_string_tuple(
                    item.get("stall_reasons"),
                    f"{path}.transfer_intervals[{index}].stall_reasons",
                ),
            )
            for index, raw in enumerate(transfers)
            for item in (_mapping(raw, f"{path}.transfer_intervals[{index}]"),)
        ),
        device_peaks=tuple(
            DeviceMemoryPeak(
                device_id=_string(
                    item.get("device_id"), f"{path}.device_peaks[{index}].device_id"
                ),
                object_bytes=_integer(
                    item.get("object_bytes"),
                    f"{path}.device_peaks[{index}].object_bytes",
                ),
                workspace_bytes=_integer(
                    item.get("workspace_bytes"),
                    f"{path}.device_peaks[{index}].workspace_bytes",
                ),
                total_bytes=_integer(
                    item.get("total_bytes"), f"{path}.device_peaks[{index}].total_bytes"
                ),
            )
            for index, raw in enumerate(peaks)
            for item in (_mapping(raw, f"{path}.device_peaks[{index}]"),)
        ),
        spill_peak_bytes=_integer(
            data.get("spill_peak_bytes"), f"{path}.spill_peak_bytes"
        ),
        memory_timeline=tuple(
            MemorySnapshot(
                time_ns=_integer(
                    item.get("time_ns"), f"{path}.memory_timeline[{index}].time_ns"
                ),
                device_object_bytes=_integer_pairs(
                    item.get("device_object_bytes"),
                    f"{path}.memory_timeline[{index}].device_object_bytes",
                ),
                device_workspace_bytes=_integer_pairs(
                    item.get("device_workspace_bytes"),
                    f"{path}.memory_timeline[{index}].device_workspace_bytes",
                ),
                spill_bytes=_integer(
                    item.get("spill_bytes"),
                    f"{path}.memory_timeline[{index}].spill_bytes",
                ),
                device_physical_bytes=_integer_pairs(
                    item.get("device_physical_bytes"),
                    f"{path}.memory_timeline[{index}].device_physical_bytes",
                ),
            )
            for index, raw in enumerate(timeline)
            for item in (_mapping(raw, f"{path}.memory_timeline[{index}]"),)
        ),
    )
