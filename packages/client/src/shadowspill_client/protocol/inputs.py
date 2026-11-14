"""Explicit tensor-tree alignment and capacity declarations; no model imports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .._records import Record, Value, freeze, integer, names, number, record, text

DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    }
)


def shape(value: Sequence[int | str]) -> tuple[int | str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("shape must be a list or tuple")
    for item in value:
        if type(item) is str:
            text(item, "dimension")
        else:
            integer(item, "dimension")
    return tuple(value)


@dataclass(frozen=True)
class TensorField(Record):
    """One leaf in a Datum tree, before adding any internal microbatch axes.

    Repeated symbolic dimensions explicitly declare aligned positions, including
    across model/loss trees. Matching numeric dimensions alone do not. A packed
    field names its packing axis; stack adds an example axis, and shared requires
    equal values within a physical batch. Defaults are broadcast only as declared.
    """

    path: Sequence[str]
    dtype: str
    shape: Sequence[int | str] = ()
    batching: Literal["stack", "pack", "shared"] = "stack"
    pack_axis: int | None = None
    required: bool = True
    default: Value = None
    padding: Value = 0
    finite: bool = True
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        # A path can repeat a component (e.g. encoder.encoder.value).
        if not isinstance(self.path, (tuple, list)) or not self.path:
            raise ValueError("field path must be a nonempty sequence")
        object.__setattr__(self, "path", tuple(text(p, "path") for p in self.path))
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported portable dtype: {self.dtype}")
        object.__setattr__(self, "shape", shape(self.shape))
        if self.batching not in ("stack", "pack", "shared"):
            raise ValueError("batching must be stack, pack or shared")
        if self.batching == "pack":
            axis = integer(self.pack_axis, "pack_axis")
            if axis >= len(self.shape):
                raise ValueError("pack_axis must index the field shape")
        elif self.pack_axis is not None:
            raise ValueError("only packed fields have a pack_axis")
        if type(self.required) is not bool or type(self.finite) is not bool:
            raise TypeError("required and finite must be booleans")
        if self.required and self.default is not None:
            raise ValueError("a required field cannot have an omission default")
        if not self.required and self.default is None:
            raise ValueError("an optional tensor field requires a default")
        for name in ("default", "padding"):
            value = freeze(getattr(self, name), name=name)
            if isinstance(value, (tuple, dict)) or (
                value is not None and type(value) not in (bool, int, float)
            ):
                raise ValueError(f"{name} must be a numeric scalar")
            if name == "padding" and value is None:
                raise ValueError("padding must be a numeric scalar")
            if value is not None:
                if self.dtype == "bool":
                    if value not in (False, True):
                        raise ValueError(f"{name} must be boolean for a boolean field")
                    value = bool(value)
                elif "int" in self.dtype:
                    bits = int(self.dtype.lstrip("uint"))
                    lower = 0 if self.dtype.startswith("u") else -(2 ** (bits - 1))
                    upper = (
                        2**bits - 1
                        if self.dtype.startswith("u")
                        else 2 ** (bits - 1) - 1
                    )
                    if type(value) is not int or not lower <= value <= upper:
                        raise ValueError(f"{name} must fit {self.dtype}")
                else:
                    number(value, name)
            object.__setattr__(self, name, value)
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if value is not None:
                number(value, name)
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum exceeds maximum")


@dataclass(frozen=True)
class Capacity(Record):
    """A schema-owned global limit, counted once per logical input partition.

    `count` counts Datum records. Other reductions measure a declared field axis.
    Scheduled capacities count padding too; replicated physical copies do not
    multiply the same logical positions. Actual routing/packing lives in engine.
    """

    name: str
    unit: str
    reduction: Literal["sum", "max", "count"]
    path: Sequence[str] = ()
    axis: int | None = None
    scheduled: bool = False

    def __post_init__(self) -> None:
        text(self.name, "capacity name")
        text(self.unit, "capacity unit")
        if self.reduction not in ("sum", "max", "count"):
            raise ValueError("capacity reduction must be sum, max or count")
        if not isinstance(self.path, (list, tuple)):
            raise TypeError("capacity path must be a sequence")
        object.__setattr__(self, "path", tuple(text(p, "path") for p in self.path))
        if self.reduction == "count":
            if self.path or self.axis is not None:
                raise ValueError("Datum count has no field or axis")
        elif not self.path:
            raise ValueError("axis capacities require a field path")
        else:
            integer(self.axis, "capacity axis")
        if type(self.scheduled) is not bool:
            raise TypeError("scheduled must be a boolean")


@dataclass(frozen=True)
class InputSchema(Record):
    fields: Sequence[TensorField]
    capacities: Sequence[Capacity] = ()

    def __post_init__(self) -> None:
        fields = tuple(record(TensorField, item, "field") for item in self.fields)
        paths = [tuple(item.path) for item in fields]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate input field path")
        if any(p != q and q[: len(p)] == p for p in paths for q in paths):
            raise ValueError("an input field cannot also be a tree parent")
        capacities = tuple(
            record(Capacity, item, "capacity") for item in self.capacities
        )
        names([item.name for item in capacities], "capacities")
        by_path = {tuple(item.path): item for item in fields}
        for item in capacities:
            if item.reduction != "count":
                field = by_path.get(tuple(item.path))
                if field is None or item.axis is None or item.axis >= len(field.shape):
                    raise ValueError(f"capacity {item.name} must name a declared axis")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "capacities", capacities)
