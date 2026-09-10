"""What a caller asks the planner for, and how an option record is written."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from fractions import Fraction
from typing import Any, ClassVar, Self


class InitialPlacement(StrEnum):
    """How host-origin objects may be placed before the first task."""

    REQUIRED = "required"
    GREEDY = "greedy"


class OptionRecord:
    """Serialization shared by every option record.

    Both halves are derived from the dataclass's own fields rather than a
    written-out list, so an option added later is carried by this and by
    everything built on it -- the plan key included -- without a second
    edit. Subclasses are frozen slotted dataclasses; this base holds no
    state of its own.

    Each subclass names a `KIND`, and records itself under it. A nested
    record therefore says on the wire what it is, and reads back as the
    type it was written as -- which is what lets a second search store its
    own options inside a `SearchOptions` without this module hearing about
    it.
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
        """Rebuild a nested record from what :meth:`to_dict` wrote."""

        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("an option record must be an object")
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in OptionRecord._KINDS:
            raise ValueError(f"unknown option record kind {kind!r}")
        return OptionRecord._KINDS[kind].from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Every option, in declaration order, as JSON-compatible values."""

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
        """Rebuild what :meth:`to_dict` wrote. Every option must be present.

        A record missing an option is a record from a different version of
        this type, and reading it as though the absent options held their
        current defaults is how a replay silently plans a different problem
        than the run it replays.
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
    """What every search understands, whichever search it is.

    Held apart from any one search's own options so that adding a search
    changes nothing here, and so a caller can see at a glance which half of
    a request is universal.

    How much of the machine to spend is not here: that is
    `SearchOptions.workers`, because it changes how long an answer takes
    rather than which answer is right.

    Every field here is part of a planned program's identity: change one
    and the question changes, so the answer is keyed separately.
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
        if self.minimum_object_bytes_evict_eligible < 0:
            raise ValueError("minimum_object_bytes_evict_eligible is invalid")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")


__all__ = ["GenericPlanningOptions", "InitialPlacement", "OptionRecord"]
