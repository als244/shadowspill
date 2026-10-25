"""A fixed-layout certificate on disk, beside the plan it certifies.

What grows with the step -- every placement, reuse dependency and
task-allocation lease of the layout, and every task and transfer interval of
the simulation that proved it -- is stored as arrays, written and read
without a Python object per entry. The rest -- digests, byte counts, the
short tables, the simulator input, the simulation's peaks and violations -- is
one JSON member, `meta`, in the forms `to_dict` and the simulation encoders
give it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shadowspill.schema import artifact_schema
from shadowspill.simulator.indexing.results import (
    TASK_COLUMNS,
    TRANSFER_COLUMNS,
    IntervalNames,
    interval_columns,
    intervals_from_columns,
)
from shadowspill.store import atomic_write

from .physical_layout import _fixed_layout_from_value
from .simulation_admission import (
    _simulation_admission_from_value,
    simulation_admission_value,
)
from .simulation_result import _simulation_result_from_value, simulation_result_value

if TYPE_CHECKING:
    from shadowspill.planner.admission.layout.model import FixedLayoutAdmission

#: The certificate's file name, beside the plan record it certifies.
CERTIFICATE_FILE = "certificate.npz"

_SCHEMA = artifact_schema("fixed_layout_certificate")
_INTERVAL_TYPES = {
    **{f"tasks.{name}": np.dtype(kind) for name, kind in TASK_COLUMNS},
    **{f"transfers.{name}": np.dtype(kind) for name, kind in TRANSFER_COLUMNS},
}
_INTERVAL_COLUMNS = tuple(_INTERVAL_TYPES)


def _groups() -> dict[str, tuple[str, ...]]:
    from shadowspill.planner.admission.layout.model import (
        FixedLayoutPlacements,
        FixedLayoutReuses,
        FixedLayoutTaskAllocations,
    )

    return {
        "placements": FixedLayoutPlacements.COLUMNS,
        "reuse_dependencies": FixedLayoutReuses.COLUMNS,
        "task_allocation_leases": FixedLayoutTaskAllocations.COLUMNS,
    }


def write_certificate(path: Path, admission: FixedLayoutAdmission) -> None:
    """Write `admission` to `path`, whole or not at all."""

    layout = admission.layout
    columns, names = interval_columns(admission.simulation)
    meta = {
        "schema": _SCHEMA,
        "facts_digest": layout.facts_digest,
        "layout": layout.header(),
        "simulator_input": simulation_admission_value(admission.simulator_input),
        "simulation": {
            **simulation_result_value(
                replace(admission.simulation, task_intervals=(), transfer_intervals=())
            ),
            "interval_names": {
                "task_ids": list(names.task_ids),
                "alias_ids": list(names.alias_ids),
                "device_ids": list(names.device_ids),
            },
        },
    }
    arrays: dict[str, np.ndarray] = {
        "meta": np.frombuffer(
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode(),
            dtype=np.uint8,
        )
    }
    for group, column_names in _groups().items():
        values = getattr(layout, group)
        for name in column_names:
            arrays[f"{group}.{name}"] = getattr(values, name)
    for name in _INTERVAL_COLUMNS:
        arrays[f"simulation.{name}"] = columns[name]
    atomic_write(path, lambda output: np.savez(output, allow_pickle=False, **arrays))


def read_certificate_meta(path: Path) -> dict[str, object]:
    """Only the `meta` member: everything but the arrays, for a reader that
    wants the byte counts or the digests and not a million entries."""

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["meta"].tobytes())
    if not isinstance(meta, dict) or meta.get("schema") != _SCHEMA:
        raise ValueError(f"{path}: unsupported certificate schema")
    return meta


def read_certificate(path: Path) -> FixedLayoutAdmission:
    """The certificate stored at `path`."""

    from shadowspill.planner.admission.layout.model import FixedLayoutAdmission

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["meta"].tobytes())
        if not isinstance(meta, dict) or meta.get("schema") != _SCHEMA:
            raise ValueError(f"{path}: unsupported certificate schema")
        layout_value = dict(meta["layout"])
        for group, column_names in _groups().items():
            layout_value[group] = {
                name: data[f"{group}.{name}"] for name in column_names
            }
        columns = {name: data[f"simulation.{name}"] for name in _INTERVAL_COLUMNS}
    layout = _fixed_layout_from_value(layout_value, f"{path}.layout")
    simulation_value = dict(meta["simulation"])
    table = simulation_value.pop("interval_names")
    names = IntervalNames(
        tuple(table["task_ids"]), tuple(table["alias_ids"]), tuple(table["device_ids"])
    )
    for name, values in columns.items():
        if values.ndim != 1 or values.dtype != _INTERVAL_TYPES[name]:
            raise ValueError(
                f"{path}: simulation column {name} is not one row of"
                f" {_INTERVAL_TYPES[name]}"
            )
    for record in ("tasks", "transfers"):
        lengths = {
            len(values)
            for name, values in columns.items()
            if name.startswith(f"{record}.")
        }
        if len(lengths) > 1:
            raise ValueError(f"{path}: the {record} columns differ in length")
    try:
        task_intervals, transfer_intervals = intervals_from_columns(columns, names)
    except (IndexError, KeyError) as exc:
        raise ValueError(
            f"{path}: a simulation column holds an index or code nothing names"
        ) from exc
    simulation = replace(
        _simulation_result_from_value(simulation_value, f"{path}.simulation"),
        task_intervals=task_intervals,
        transfer_intervals=transfer_intervals,
    )
    return FixedLayoutAdmission(
        layout=layout,
        simulator_input=_simulation_admission_from_value(
            meta["simulator_input"], f"{path}.simulator_input"
        ),
        simulation=simulation,
    )


__all__ = [
    "CERTIFICATE_FILE",
    "read_certificate",
    "read_certificate_meta",
    "write_certificate",
]
