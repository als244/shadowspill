"""What a caller asks the planner for, and how an option record is written."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar, Self


class OptionRecord:
    """The base of every option record, and how one is written and read.

    A subclass is a frozen, slotted dataclass whose fields are its options.
    `OptionRecord.to_dict` and `OptionRecord.from_dict` follow its fields,
    so an option added later reaches the plan key and the archived request
    with no other change. A subclass that names a ``KIND`` is registered
    under it when it is defined, so a record nested in another reads back
    as the type it was written as; that is how a search stores its own
    options inside a `SearchOptions`.

    Attributes
    ----------
    KIND : str
        The name the record is written and registered under, set by the
        subclass. Empty on the base, which writes no ``kind``.
    """

    __slots__ = ()

    KIND: ClassVar[str] = ""
    _KINDS: ClassVar[dict[str, type[OptionRecord]]] = {}

    def __init_subclass__(cls, **keywords: object) -> None:
        super().__init_subclass__(**keywords)
        if cls.KIND:
            OptionRecord._KINDS[cls.KIND] = cls

    @staticmethod
    def record_from_value(value: object) -> OptionRecord | None:
        """Read a record back as the type its ``kind`` names.

        Parameters
        ----------
        value : object
            What `OptionRecord.to_dict` wrote, or None.

        Returns
        -------
        OptionRecord or None
            The record, an instance of the registered subclass; None when
            ``value`` is None.

        Raises
        ------
        ValueError
            If ``value`` is not a mapping, its ``kind`` names no registered
            record, or the record misses an option.
        """

        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("an option record must be an object")
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in OptionRecord._KINDS:
            raise ValueError(f"unknown option record kind {kind!r}")
        return OptionRecord._KINDS[kind].from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Write every option, in declaration order, as JSON-compatible values.

        A nested record is written as its own mapping, an enumeration as its
        value, and a tuple as a list with fractions as strings.

        Returns
        -------
        dict
            The options by name, after ``kind`` when the record names one.
        """

        record: dict[str, Any] = {"kind": self.KIND} if self.KIND else {}
        for option in fields(self):  # type: ignore[arg-type]
            value = getattr(self, option.name)
            if isinstance(value, OptionRecord):
                value = value.to_dict()
            elif isinstance(value, Enum):
                value = value.value
            elif isinstance(value, tuple):
                value = [
                    str(item) if isinstance(item, Fraction) else item for item in value
                ]
            record[option.name] = value
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Read a record of this type back from what `OptionRecord.to_dict` wrote.

        Every option must be present: a record missing one comes from
        another version of the type, and filling it from the current default
        would replay a different problem.

        Parameters
        ----------
        record : mapping of str to object
            The written options.

        Returns
        -------
        OptionRecord
            An instance of this class.

        Raises
        ------
        ValueError
            If an option is missing.
        """

        missing = sorted(
            option.name
            for option in fields(cls)  # type: ignore[arg-type]
            if option.name not in record
        )
        if missing:
            raise ValueError(f"options omit {', '.join(missing)}")
        values: dict[str, Any] = {}
        for option in fields(cls):  # type: ignore[arg-type]
            value = record[option.name]
            default = option.default
            if isinstance(default, Enum):
                value = type(default)(value)
            elif isinstance(default, tuple):
                if default and isinstance(default[0], Fraction):
                    value = tuple(Fraction(item) for item in value)
                else:
                    value = tuple(value)
            values[option.name] = value
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GenericPlanningOptions(OptionRecord):
    """The planning options every search is told, whichever search runs.

    Both are part of the plan key; `SearchOptions.workers` is not.

    Parameters
    ----------
    deterministic : bool, default False
        Make every candidate's outcome depend only on its inputs, so any
        worker count gives the same answer. The placement gate then consults
        only the candidate's own placed plans instead of the best plan any
        worker has placed, which costs more placement measurements.
    minimum_object_bytes_evict_eligible : int, default 1048576
        Objects smaller than this many bytes (1 MiB by default) are never
        evicted mid-step: they stay resident from their first access to
        their last, each lease with a fixed home in the result's
        `ResidentSlice`; their fetches, release and write-back are scheduled
        as for any object. At least zero and below 2**64; zero makes every object
        eligible.

    Raises
    ------
    ValueError
        If ``minimum_object_bytes_evict_eligible`` is negative or not below
        2**64, or ``deterministic`` is not a bool.
    """

    KIND: ClassVar[str] = "generic"

    #: Make every candidate's outcome a pure function of its inputs, so
    #: parallel planning reproduces exactly run to run. The placement gate
    #: then consults only the candidate's own placed plans, never the shared
    #: best-placed record, which costs additional placement measurements.
    #: Off by default: the shared gate is faster, and the default search is
    #: stable without being bit-reproducible.
    deterministic: bool = False
    #: Objects smaller than this many bytes are not eligible to be evicted
    #: mid-step: they stay resident from their first to their last access,
    #: and the planner charges them at every boundary in between. Their
    #: boundary contract is untouched -- an opening fetch, a release after
    #: the last access, a terminal writeback when modified. The default is
    #: 1 MiB, because a copy under that size is latency-bound and its bytes
    #: hardly relieve a boundary, while every such object is a cut
    #: candidate, a dispatch, and an event. Zero makes every object
    #: eligible, which is what a caller planning byte-sized objects wants.
    minimum_object_bytes_evict_eligible: int = 1 << 20

    def __post_init__(self) -> None:
        # The planner reads it as a 64-bit count, which would wrap.
        if not 0 <= self.minimum_object_bytes_evict_eligible < 1 << 64:
            raise ValueError("minimum_object_bytes_evict_eligible is invalid")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")


__all__ = ["GenericPlanningOptions", "OptionRecord"]
