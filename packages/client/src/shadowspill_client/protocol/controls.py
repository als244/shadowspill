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
    """The declaration of one per-call control: its type, default and range.

    Parameters
    ----------
    default : bool, int or float
        Value used when neither the run's defaults nor the call give one.
    dtype : str, default "float"
        ``"float"`` (any finite int or float), ``"int"`` or ``"bool"``.
    minimum : float, optional
        Lowest allowed value; no lower bound by default.
    maximum : float, optional
        Highest allowed value; no upper bound by default.
    minimum_inclusive : bool, default True
        Whether ``minimum`` itself is allowed.
    maximum_inclusive : bool, default True
        Whether ``maximum`` itself is allowed.

    Raises
    ------
    TypeError
        If an inclusivity flag is not a bool.
    ValueError
        If ``dtype`` is unknown, a bound is not a finite number, a ``"bool"``
        control has bounds, or ``default`` is not an allowed value.
    """

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
        """Check one value against this declaration.

        Parameters
        ----------
        value : bool, int or float
            The value to check.

        Returns
        -------
        bool, int or float
            ``value``, unchanged.

        Raises
        ------
        ValueError
            If ``value`` has the wrong type, is not finite, or lies outside the
            bounds.
        """
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
    """The named per-call controls an optimizer or objective accepts.

    An optimizer's schema declares controls such as ``"lr"``. A call's
    ``optim`` or ``loss_params`` values are checked against it, and do not
    carry over to later calls.

    Parameters
    ----------
    fields : mapping of str to ScalarControl
        Each control's declaration, by name; kept as a read-only mapping.

    Raises
    ------
    TypeError
        If ``fields`` is not a mapping, or a declaration is neither a
        `ScalarControl` nor a mapping of its fields.
    ValueError
        If a name is empty or has surrounding spaces, or a declaration is
        invalid.
    """

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
        """Resolve every control's value for one call.

        Each control starts at its declared default; ``defaults`` replaces
        those, and ``overrides`` replaces both. Every value in both layers is
        checked, so an override cannot hide an invalid default.

        Parameters
        ----------
        defaults : mapping of str to scalar, optional
            The run's defaults, such as ``Optimizer(defaults=...)``.
        overrides : mapping of str to scalar, optional
            The call's values, such as a step's ``optim``.

        Returns
        -------
        mapping of str to scalar
            One value per declared control, read-only.

        Raises
        ------
        TypeError
            If a layer is not a mapping.
        ValueError
            If a layer names an undeclared control or holds an invalid value.
        """
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
