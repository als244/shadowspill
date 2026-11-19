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
    """One field of a `Datum` tree, as a model or objective declares it.

    ``shape`` is the shape of one example's value. A symbolic dimension (a
    str) takes its size from the data; the same name, in this field or any
    other of the model's and the objective's schemas, must have the same size
    within one example. Equal integer sizes alone imply no alignment.

    Parameters
    ----------
    path : sequence of str
        Keys from the top of the tree to the field, such as ``("tokens",)``.
    dtype : str
        Element type: ``"bool"``, ``"uint8"``, ``"int8"``, ``"int16"``,
        ``"int32"``, ``"int64"``, ``"float16"``, ``"bfloat16"``,
        ``"float32"`` or ``"float64"``. Supplied values must be representable
        in it, and are converted to it when packed.
    shape : sequence of int or str, default ()
        Dimensions of one example's value, each fixed (int) or symbolic (str).
    batching : str, default "stack"
        How examples share a physical buffer: ``"stack"`` adds a leading
        example axis, ``"pack"`` concatenates examples along ``pack_axis``,
        and ``"shared"`` keeps one value per buffer, which every example in it
        must supply equally.
    pack_axis : int, optional
        Axis of ``shape`` along which packed examples are concatenated.
        Required for ``"pack"``, None otherwise.
    required : bool, default True
        Whether every example must supply the field when an operation reads
        it.
    default : bool, int or float, optional
        Value of an omitted optional field, broadcast to its shape. Required
        for an optional field, None for a required one.
    padding : bool, int or float, default 0
        Value that fills padding in a physical buffer, and the positions
        ``validity`` marks inactive.
    finite : bool, default True
        Whether every value must be finite.
    minimum : float, optional
        Smallest allowed value, inclusive. None for no bound.
    maximum : float, optional
        Largest allowed value, inclusive. None for no bound.
    validity : sequence of str, optional
        Path of a boolean field in the same tree with the same shape and
        batching. When the operation reads that mask, only the positions it
        marks true have their values checked, and the others are replaced by
        ``padding``; otherwise every position is checked.

    Raises
    ------
    ValueError
        The declaration is inconsistent: an empty path, an unknown dtype or
        batching, a ``pack_axis`` missing for ``"pack"`` or given otherwise,
        a default given for a required field or missing for an optional one,
        a default or padding that is not a scalar of the dtype, ``minimum``
        above ``maximum``, or ``padding`` outside the bounds of a masked
        field.
    TypeError
        ``shape`` is not a list or tuple, or ``required`` or ``finite`` is not
        a bool.
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
    validity: Sequence[str] | None = None

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
        if self.validity is not None:
            if not isinstance(self.validity, (tuple, list)) or not self.validity:
                raise ValueError("validity must name a boolean field in the same tree")
            object.__setattr__(
                self, "validity", tuple(text(p, "validity path") for p in self.validity)
            )
            assert isinstance(self.padding, (bool, int, float))
            if (self.minimum is not None and self.padding < self.minimum) or (
                self.maximum is not None and self.padding > self.maximum
            ):
                raise ValueError("masked field padding must satisfy its numeric bounds")


@dataclass(frozen=True)
class Capacity(Record):
    """A limit on a batch, as an input schema declares it.

    A `BatchConfig` gives each limit's value under its name, in ``limits``.
    The limit applies to the whole batch, over every input partition; copies
    of the same examples on cooperating workers count once.

    Parameters
    ----------
    name : str
        The limit's name in ``BatchConfig.limits``, such as
        ``"token_capacity"``.
    unit : str
        Unit of the counted quantity, such as ``"tokens"``. Plan reports
        count tokens per second from the one scheduled capacity in
        ``"tokens"``, when there is exactly one.
    reduction : str
        What is counted: ``"count"`` counts examples, ``"sum"`` adds the size
        of ``axis`` over every example, and ``"max"`` takes its largest size.
    path : sequence of str, default ()
        Path of the measured field, in the same tree as the schema's fields.
        Empty for ``"count"``.
    axis : int, optional
        Axis of the field's ``shape`` that is measured. None for ``"count"``.
    scheduled : bool, default False
        Whether the limit also bounds each prepared packing schedule, whose
        totals include padding.

    Raises
    ------
    ValueError
        ``name`` or ``unit`` is empty, ``reduction`` is unknown, or ``path``
        and ``axis`` are given for ``"count"`` or missing otherwise.
    TypeError
        ``path`` is not a list or tuple, or ``scheduled`` is not a bool.
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
    """The input fields and limits a model or objective declares.

    A model's schema describes `Datum.inputs`, and an objective's describes
    `Datum.loss_inputs`. Each submission is checked against both before it
    uploads.

    Parameters
    ----------
    fields : sequence of TensorField
        The fields.
    capacities : sequence of Capacity, default ()
        The limits.

    Raises
    ------
    ValueError
        Two fields share a path, one field's path extends another's, two
        capacities share a name, a ``validity`` path does not name an
        unmasked boolean field with the same shape and batching, or a
        capacity measures an undeclared field or axis.
    """

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
        for declared in fields:
            if declared.validity is None:
                continue
            mask = by_path.get(tuple(declared.validity))
            if (
                mask is None
                or mask.dtype != "bool"
                or mask.validity is not None
                or mask.shape != declared.shape
                or mask.batching != declared.batching
                or mask.pack_axis != declared.pack_axis
            ):
                raise ValueError(
                    "validity requires an unmasked boolean field "
                    "with identical alignment"
                )
        for item in capacities:
            if item.reduction != "count":
                field = by_path.get(tuple(item.path))
                if field is None or item.axis is None or item.axis >= len(field.shape):
                    raise ValueError(f"capacity {item.name} must name a declared axis")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "capacities", capacities)
