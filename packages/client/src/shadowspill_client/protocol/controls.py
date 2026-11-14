"""Declared runtime scalar domains and nonsticky default resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .._records import Record, number, record, text

type Scalar = bool | int | float


@dataclass(frozen=True)
class ScalarControl(Record):
    default: Scalar
    dtype: Literal["float", "int", "bool"] = "float"
    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in ("float", "int", "bool"):
            raise ValueError("control dtype must be float, int or bool")
        if any(
            type(v) is not bool
            for v in (self.minimum_inclusive, self.maximum_inclusive)
        ):
            raise TypeError("control bound inclusivity must be boolean")
        for bound in (self.minimum, self.maximum):
            if bound is not None:
                number(bound, "control bound")
        if self.dtype == "bool" and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("boolean controls do not have numeric bounds")
        self.validate(self.default)

    def validate(self, value: Scalar) -> Scalar:
        if self.dtype == "bool":
            if type(value) is not bool:
                raise ValueError("control requires a boolean")
            return value
        if self.dtype == "int" and type(value) is not int:
            raise ValueError("control requires an integer")
        numeric = number(value, "control")
        if self.minimum is not None and (
            numeric < self.minimum
            or (numeric == self.minimum and not self.minimum_inclusive)
        ):
            raise ValueError("control is below its allowed minimum")
        if self.maximum is not None and (
            numeric > self.maximum
            or (numeric == self.maximum and not self.maximum_inclusive)
        ):
            raise ValueError("control exceeds its allowed maximum")
        return value


@dataclass(frozen=True)
class ControlSchema(Record):
    fields: Mapping[str, ScalarControl]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping):
            raise TypeError("controls must be a mapping")
        fields = {
            text(k, "control name"): record(ScalarControl, v, k)
            for k, v in self.fields.items()
        }
        object.__setattr__(self, "fields", MappingProxyType(fields))

    def resolve(
        self,
        *,
        defaults: Mapping[str, Scalar] | None = None,
        overrides: Mapping[str, Scalar] | None = None,
    ) -> Mapping[str, Scalar]:
        """Validate both layers; a later override cannot hide an invalid default."""
        result = {name: field.default for name, field in self.fields.items()}
        for layer in (defaults, overrides):
            if layer is None:
                continue
            if not isinstance(layer, Mapping):
                raise TypeError("controls must be a mapping")
            if unknown := layer.keys() - self.fields.keys():
                raise ValueError(f"unknown controls: {sorted(unknown)}")
            for name, value in layer.items():
                result[name] = self.fields[name].validate(value)
        return MappingProxyType(result)
