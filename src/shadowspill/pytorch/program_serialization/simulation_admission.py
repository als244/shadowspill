"""Serialization for simulator physical-admission inputs."""

from __future__ import annotations

from shadowspill.simulator import (
    ActionPhysicalDelta,
    MemoryReuseDependency,
    SimulationAdmission,
    TaskPhysicalDelta,
)

from .common import (
    _integer,
    _integer_pairs,
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _string,
)


def _simulation_admission_from_value(
    value: object,
    path: str,
) -> SimulationAdmission:
    data = _mapping(value, path)
    task_deltas = _list(data.get("task_deltas"), f"{path}.task_deltas")
    action_deltas = _list(data.get("action_deltas"), f"{path}.action_deltas")
    dependencies = _list(data.get("reuse_dependencies"), f"{path}.reuse_dependencies")
    return SimulationAdmission(
        initial_physical_bytes=_integer_pairs(
            data.get("initial_physical_bytes"), f"{path}.initial_physical_bytes"
        ),
        device_capacity_bytes=_integer_pairs(
            data.get("device_capacity_bytes"), f"{path}.device_capacity_bytes"
        ),
        task_deltas=tuple(
            TaskPhysicalDelta(
                task_id=_string(
                    item.get("task_id"), f"{path}.task_deltas[{index}].task_id"
                ),
                start_bytes=_integer(
                    item.get("start_bytes"), f"{path}.task_deltas[{index}].start_bytes"
                ),
                completion_bytes=_integer(
                    item.get("completion_bytes"),
                    f"{path}.task_deltas[{index}].completion_bytes",
                ),
            )
            for index, raw in enumerate(task_deltas)
            for item in (_mapping(raw, f"{path}.task_deltas[{index}]"),)
        ),
        action_deltas=tuple(
            ActionPhysicalDelta(
                action_index=_integer(
                    item.get("action_index"),
                    f"{path}.action_deltas[{index}].action_index",
                ),
                trigger_bytes=_integer(
                    item.get("trigger_bytes"),
                    f"{path}.action_deltas[{index}].trigger_bytes",
                ),
                completion_bytes=_integer(
                    item.get("completion_bytes"),
                    f"{path}.action_deltas[{index}].completion_bytes",
                ),
            )
            for index, raw in enumerate(action_deltas)
            for item in (_mapping(raw, f"{path}.action_deltas[{index}]"),)
        ),
        reuse_dependencies=tuple(
            MemoryReuseDependency(
                predecessor_action_index=_integer(
                    item.get("predecessor_action_index"),
                    f"{path}.reuse_dependencies[{index}].predecessor_action_index",
                ),
                successor_task_id=_optional_string(
                    item.get("successor_task_id"),
                    f"{path}.reuse_dependencies[{index}].successor_task_id",
                ),
                successor_action_index=_optional_integer(
                    item.get("successor_action_index"),
                    f"{path}.reuse_dependencies[{index}].successor_action_index",
                ),
            )
            for index, raw in enumerate(dependencies)
            for item in (_mapping(raw, f"{path}.reuse_dependencies[{index}]"),)
        ),
    )
