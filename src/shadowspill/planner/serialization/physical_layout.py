"""Serialization for fixed physical-layout certificates."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from shadowspill.schema import artifact_schema

from .common import (
    _integer,
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _string,
)

if TYPE_CHECKING:
    from shadowspill.planner.admission.layout.model import FixedPhysicalLayout


def _fixed_layout_from_value(value: object, path: str) -> FixedPhysicalLayout:
    from shadowspill.planner.admission.admission_replay import AdmissionReplayPurpose
    from shadowspill.planner.admission.layout.model import (
        FixedLayoutPlacement,
        FixedLayoutReuse,
        FixedPhysicalLayout,
        LeaseLifetime,
    )

    data = _mapping(value, path)
    if data.get("schema") != artifact_schema("fixed_physical_layout"):
        raise ValueError(f"{path}.schema: unsupported schema")
    placements = _list(data.get("placements"), f"{path}.placements")
    dependencies = _list(data.get("reuse_dependencies"), f"{path}.reuse_dependencies")
    dynamic = _list(data.get("dynamic_lifetimes"), f"{path}.dynamic_lifetimes")
    initial = _list(data.get("initial_alias_leases"), f"{path}.initial_alias_leases")
    allocations = _list(
        data.get("task_allocation_leases"), f"{path}.task_allocation_leases"
    )
    actions = _list(
        data.get("action_destination_leases"),
        f"{path}.action_destination_leases",
    )

    def lifetime(item: dict[str, Any], item_path: str) -> LeaseLifetime:
        return LeaseLifetime(
            lease_id=_integer(item.get("lease_id"), f"{item_path}.lease_id"),
            bytes=_integer(item.get("bytes"), f"{item_path}.bytes"),
            alignment=_integer(item.get("alignment"), f"{item_path}.alignment"),
            predicted_start_ns=_integer(
                item.get("predicted_start_ns"), f"{item_path}.predicted_start_ns"
            ),
            predicted_end_ns=_integer(
                item.get("predicted_end_ns"), f"{item_path}.predicted_end_ns"
            ),
            causal_start=_integer(
                item.get("causal_start"), f"{item_path}.causal_start"
            ),
            causal_end=_integer(item.get("causal_end"), f"{item_path}.causal_end"),
            purpose=AdmissionReplayPurpose(
                _string(item.get("purpose"), f"{item_path}.purpose")
            ),
            task_id=_optional_string(item.get("task_id"), f"{item_path}.task_id"),
            alias_group_id=_optional_string(
                item.get("alias_group_id"), f"{item_path}.alias_group_id"
            ),
            action_index=_optional_integer(
                item.get("action_index"), f"{item_path}.action_index"
            ),
        )

    return FixedPhysicalLayout(
        program_digest=_string(data.get("program_digest"), f"{path}.program_digest"),
        schedule_digest=_string(data.get("schedule_digest"), f"{path}.schedule_digest"),
        facts_digest=_string(data.get("facts_digest"), f"{path}.facts_digest"),
        pool_capacity_bytes=_integer(
            data.get("pool_capacity_bytes"), f"{path}.pool_capacity_bytes"
        ),
        fixed_slice_bytes=_integer(
            data.get("fixed_slice_bytes"), f"{path}.fixed_slice_bytes"
        ),
        resident_slice_bytes=_integer(
            data.get("resident_slice_bytes"), f"{path}.resident_slice_bytes"
        ),
        dynamic_reserve_bytes=_integer(
            data.get("dynamic_reserve_bytes"), f"{path}.dynamic_reserve_bytes"
        ),
        scratch_reserve_bytes=_integer(
            data.get("scratch_reserve_bytes"), f"{path}.scratch_reserve_bytes"
        ),
        required_bytes=_integer(data.get("required_bytes"), f"{path}.required_bytes"),
        placements=tuple(
            FixedLayoutPlacement(
                **asdict(lifetime(item, f"{path}.placements[{index}]")),
                offset=_integer(
                    item.get("offset"), f"{path}.placements[{index}].offset"
                ),
            )
            for index, raw in enumerate(placements)
            for item in (_mapping(raw, f"{path}.placements[{index}]"),)
        ),
        reuse_dependencies=tuple(
            FixedLayoutReuse(
                dependency_id=_integer(
                    item.get("dependency_id"),
                    f"{path}.reuse_dependencies[{index}].dependency_id",
                ),
                predecessor_lease_id=_integer(
                    item.get("predecessor_lease_id"),
                    f"{path}.reuse_dependencies[{index}].predecessor_lease_id",
                ),
                predecessor_purpose=AdmissionReplayPurpose(
                    _string(
                        item.get("predecessor_purpose"),
                        f"{path}.reuse_dependencies[{index}].predecessor_purpose",
                    )
                ),
                predecessor_task_id=_string(
                    item.get("predecessor_task_id"),
                    f"{path}.reuse_dependencies[{index}].predecessor_task_id",
                ),
                predecessor_action_index=_optional_integer(
                    item.get("predecessor_action_index"),
                    f"{path}.reuse_dependencies[{index}].predecessor_action_index",
                ),
                successor_lease_id=_integer(
                    item.get("successor_lease_id"),
                    f"{path}.reuse_dependencies[{index}].successor_lease_id",
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
        initial_alias_leases=tuple(
            (
                _string(
                    item.get("alias_group_id"),
                    f"{path}.initial_alias_leases[{index}].alias_group_id",
                ),
                _integer(
                    item.get("lease_id"),
                    f"{path}.initial_alias_leases[{index}].lease_id",
                ),
            )
            for index, raw in enumerate(initial)
            for item in (_mapping(raw, f"{path}.initial_alias_leases[{index}]"),)
        ),
        task_allocation_leases=tuple(
            (
                _string(
                    item.get("task_id"),
                    f"{path}.task_allocation_leases[{index}].task_id",
                ),
                _integer(
                    item.get("allocation_ordinal"),
                    f"{path}.task_allocation_leases[{index}].allocation_ordinal",
                ),
                _integer(
                    item.get("lease_id"),
                    f"{path}.task_allocation_leases[{index}].lease_id",
                ),
            )
            for index, raw in enumerate(allocations)
            for item in (_mapping(raw, f"{path}.task_allocation_leases[{index}]"),)
        ),
        action_destination_leases=tuple(
            (
                _integer(
                    item.get("action_index"),
                    f"{path}.action_destination_leases[{index}].action_index",
                ),
                _integer(
                    item.get("lease_id"),
                    f"{path}.action_destination_leases[{index}].lease_id",
                ),
            )
            for index, raw in enumerate(actions)
            for item in (_mapping(raw, f"{path}.action_destination_leases[{index}]"),)
        ),
        dynamic_lifetimes=tuple(
            lifetime(item, f"{path}.dynamic_lifetimes[{index}]")
            for index, raw in enumerate(dynamic)
            for item in (_mapping(raw, f"{path}.dynamic_lifetimes[{index}]"),)
        ),
    )
