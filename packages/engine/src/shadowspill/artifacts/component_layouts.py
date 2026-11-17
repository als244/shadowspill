"""Explicit parameter-axis coordinates for stored tensor representation leaves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .state import StateId

if TYPE_CHECKING:
    from .program import Program


@dataclass(frozen=True)
class ComponentLayout:
    """A component's axes select parameter axes, without transforming its values.

    The author declares the interpretation; equal sizes alone do not establish
    it. ``encoding`` versions that interpretation across checkpoint layouts.
    ``blocks`` gives the number of parameter elements covered by one stored
    component value on each selected axis. Ownership boundaries must align to
    those blocks; this maps scale coordinates without converting their values.
    """

    parameter: str
    component_path: tuple[str, ...]
    axes: tuple[int, ...]
    encoding: str
    blocks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        identity = StateId("parameter", self.parameter, self.component_path)
        if not identity.component_path:
            raise ValueError("component layout requires a representation component")
        if (
            not isinstance(self.axes, (list, tuple))
            or any(type(axis) is not int or axis < 0 for axis in self.axes)
            or len(set(self.axes)) != len(self.axes)
        ):
            raise ValueError("component axes must be distinct nonnegative integers")
        if (
            not isinstance(self.encoding, str)
            or not self.encoding
            or "\x00" in self.encoding
        ):
            raise ValueError("component layout requires an encoding identity")
        object.__setattr__(self, "component_path", identity.component_path)
        object.__setattr__(self, "axes", tuple(self.axes))
        blocks = tuple(self.blocks) or (1,) * len(self.axes)
        if len(blocks) != len(self.axes) or any(
            type(size) is not int or size < 1 for size in blocks
        ):
            raise ValueError(
                "component blocks must be positive integers for every axis"
            )
        object.__setattr__(self, "blocks", blocks)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            parameter=self.parameter,
            component_path=list(self.component_path),
            axes=list(self.axes),
            encoding=self.encoding,
            **(
                {"blocks": list(self.blocks)}
                if any(size != 1 for size in self.blocks)
                else {}
            ),
        )


def declared_layouts(
    values: Sequence[ComponentLayout],
) -> tuple[ComponentLayout, ...]:
    result = tuple(values)
    if any(not isinstance(value, ComponentLayout) for value in result):
        raise TypeError("component layouts require ComponentLayout descriptions")
    if len({(value.parameter, value.component_path) for value in result}) != len(
        result
    ):
        raise ValueError("duplicate component layout")
    return result


def saved_layouts(
    program: Program,
) -> Mapping[tuple[str, tuple[str, ...]], ComponentLayout]:
    """Recheck every explicit declaration against captured physical ownership."""
    values = declared_layouts(
        tuple(
            ComponentLayout(**row)
            for row in program.bindings.get("component_layouts", ())
        )
    )
    if not values:
        return {}
    registrations = {
        (row["name"], tuple(row.get("component_path", ()))): row
        for row in program.bindings.get("registrations", ())
        if row["parameter"]
    }
    owners = {
        name: row
        for row in program.bindings.get("parameter_ownership", ())
        for name in row["names"]
    }
    geometry = program.bindings.get("model_tensor_geometry", {})
    result = {}
    for value in values:
        key = value.parameter, value.component_path
        row, owner = registrations.get(key), owners.get(value.parameter)
        if row is None or owner is None:
            raise ValueError("component layout has no registered parameter owner")
        shape = tuple(owner["local_shape"])
        if any(axis >= len(shape) for axis in value.axes):
            raise ValueError("component axis exceeds the parameter rank")
        actual = geometry.get(row["object_id"])
        if actual is None or tuple(actual["shape"]) != tuple(
            (shape[axis] + block - 1) // block
            for axis, block in zip(value.axes, value.blocks, strict=True)
        ):
            raise ValueError("component layout differs from captured physical shape")
        result[key] = value
    return result


def attach_component_layouts(
    program: Program, values: Sequence[ComponentLayout]
) -> Program:
    """Bind source-declared coordinates; execution tasks/storage remain unchanged."""
    values = declared_layouts(values)
    if not values:
        return program
    result = replace(
        program,
        bindings={
            **program.bindings,
            "component_layouts": [value.to_dict() for value in values],
        },
    )
    saved_layouts(result)
    return result
