"""Serialization for fixed physical-layout certificates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

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
    """Restore one fixed physical layout from its stored record.

    One decoder reads one record. It is long because the record is wide,
    not because it does several things: every line is one key, read in the
    order the writer emits it. Splitting it would scatter one schema across
    several files and leave no part easier to check against the writer.
    """

    from shadowspill.planner.admission.admission_replay import AdmissionReplayPurpose
    from shadowspill.planner.admission.layout.model import (
        PURPOSES,
        FixedLayoutPlacements,
        FixedLayoutReuses,
        FixedLayoutTaskAllocations,
        FixedPhysicalLayout,
        LeaseLifetime,
    )

    data = _mapping(value, path)
    if data.get("schema") != artifact_schema("fixed_physical_layout"):
        raise ValueError(f"{path}.schema: unsupported schema")
    if data.get("purposes") != [item.value for item in PURPOSES]:
        raise ValueError(f"{path}.purposes: does not match the purpose codes")
    task_ids = _names(data.get("task_ids"), f"{path}.task_ids")
    alias_ids = _names(data.get("alias_ids"), f"{path}.alias_ids")
    tasks, aliases = len(task_ids), len(alias_ids)
    purposes = len(PURPOSES)
    unsigned = 1 << 64
    # Each column's element type and the range its entries must fall in;
    # -1 is `NONE`, for an entry that names nothing.
    placement_columns = _columns(
        data.get("placements"),
        f"{path}.placements",
        FixedLayoutPlacements.COLUMNS,
        {
            "lease_id": (np.uint64, 0, unsigned),
            "offset": (np.uint64, 0, unsigned),
            "bytes": (np.uint64, 0, unsigned),
            "alignment": (np.uint64, 0, unsigned),
            "predicted_start_ns": (np.uint64, 0, unsigned),
            "predicted_end_ns": (np.uint64, 0, unsigned),
            "causal_start": (np.uint64, 0, unsigned),
            "causal_end": (np.uint64, 0, unsigned),
            "purpose": (np.uint8, 0, purposes),
            "task": (np.int32, -1, tasks),
            "alias": (np.int32, -1, aliases),
            "action": (np.int32, -1, 1 << 31),
        },
    )
    reuse_columns = _columns(
        data.get("reuse_dependencies"),
        f"{path}.reuse_dependencies",
        FixedLayoutReuses.COLUMNS,
        {
            "dependency_id": (np.uint64, 0, unsigned),
            "predecessor_lease_id": (np.uint64, 0, unsigned),
            "predecessor_purpose": (np.uint8, 0, purposes),
            "predecessor_task": (np.int32, 0, tasks),
            "predecessor_action": (np.int32, -1, 1 << 31),
            "successor_lease_id": (np.uint64, 0, unsigned),
            "successor_task": (np.int32, -1, tasks),
            "successor_action": (np.int32, -1, 1 << 31),
        },
    )
    allocation_columns = _columns(
        data.get("task_allocation_leases"),
        f"{path}.task_allocation_leases",
        FixedLayoutTaskAllocations.COLUMNS,
        {
            "task": (np.int32, 0, tasks),
            "allocation_ordinal": (np.uint32, 0, 1 << 32),
            "lease_id": (np.uint64, 0, unsigned),
        },
    )
    dynamic = _list(data.get("dynamic_lifetimes"), f"{path}.dynamic_lifetimes")
    initial = _list(data.get("initial_alias_leases"), f"{path}.initial_alias_leases")
    actions = _list(
        data.get("action_destination_leases"),
        f"{path}.action_destination_leases",
    )

    def lifetime_fields(item: dict[str, Any], item_path: str) -> dict[str, Any]:
        """A lease lifetime's fields, for the lifetime or the placement built on it."""

        return dict(
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
        placements=FixedLayoutPlacements(
            **placement_columns, task_ids=task_ids, alias_ids=alias_ids
        ),
        reuse_dependencies=FixedLayoutReuses(**reuse_columns, task_ids=task_ids),
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
        task_allocation_leases=FixedLayoutTaskAllocations(
            **allocation_columns, task_ids=task_ids
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
            LeaseLifetime(**lifetime_fields(item, f"{path}.dynamic_lifetimes[{index}]"))
            for index, raw in enumerate(dynamic)
            for item in (_mapping(raw, f"{path}.dynamic_lifetimes[{index}]"),)
        ),
    )


def _names(value: object, path: str) -> tuple[str, ...]:
    """A name table: a list of distinct strings."""

    names = tuple(_list(value, path))
    if not all(type(item) is str for item in names):
        raise ValueError(f"{path}: expected a list of strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: repeats a name")
    return names


def _columns(
    value: object,
    path: str,
    names: tuple[str, ...],
    specs: dict[str, tuple[type[np.generic], int, int]],
) -> dict[str, np.ndarray]:
    """One column group: exactly the named columns, equally long, each a list
    of integers in its range, converted to its element type."""

    from shadowspill.planner.admission.layout.model import column

    group = _mapping(value, path)
    if set(group) != set(names):
        raise ValueError(f"{path}: expected the columns {sorted(names)}")
    columns: dict[str, np.ndarray] = {}
    for name in names:
        dtype, low, high = specs[name]
        value = group.get(name)
        if isinstance(value, np.ndarray):
            raw = value
        else:
            items = _list(value, f"{path}.{name}")
            raw = np.asarray(items) if items else np.zeros(0, dtype=np.int64)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError(f"{path}.{name}: expected a list of integers")
        if raw.size and (int(raw.min()) < low or int(raw.max()) >= high):
            raise ValueError(f"{path}.{name}: an entry is outside [{low}, {high})")
        columns[name] = column(raw, dtype)
    if len({len(item) for item in columns.values()}) > 1:
        raise ValueError(f"{path}: columns differ in length")
    return columns
