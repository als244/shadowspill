"""Physical spill sizes and ordered transfer phases for one logical alias.

Copy phases use the calibrated directional host link. Timed phases use an
explicit serial resource, for example a collective communicator. Readiness is
published only after the entire recipe completes. Runtime adapters must provide
the same transformations; a timing recipe alone does not implement a transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.serialization.sharing import shared_writer

from ..serialization import JsonValue
from ..validation import (
    expect_boolean,
    expect_integer,
    expect_list,
    expect_mapping,
    expect_string,
    field,
    require,
    require_identifier,
    require_non_negative,
    require_tuple,
)


@dataclass(frozen=True, slots=True)
class TransferPhase:
    """One step of moving an alias group, in a `TransferSpec` recipe.

    Parameters
    ----------
    kind : {"copy", "timed"}
        ``"copy"`` moves ``bytes`` over the fetch or evict route at its
        calibrated rate. ``"timed"`` occupies the serial ``resource`` for
        ``duration_ns``, such as a collective over a sharded group.
    bytes : int, default 0
        Bytes a copy moves; for timed work, the message size its overlap
        samples are looked up at.
    duration_ns : int, default 0
        How long timed work occupies its resource, in nanoseconds. Zero for a
        copy.
    resource : str, optional
        The serial resource timed work occupies. Required for timed work, None
        for a copy.
    submission_ordered : bool, default False
        Whether timed work runs in trigger order, evictions before fetches at
        one boundary. Timed work only, and at most one phase per recipe.
    timing_class : str, optional
        The class of timed work, which selects a task's `TaskOverlapCost`
        samples. Timed work only; None, or a class no profile measured, overlaps
        at full rate.

    Raises
    ------
    ValidationError
        If ``kind`` is unknown, a size or duration is negative, or a field is
        given that the kind does not take.
    """

    kind: str
    bytes: int = 0
    duration_ns: int = 0
    resource: str | None = None
    submission_ordered: bool = False
    timing_class: str | None = None

    def __post_init__(self) -> None:
        if self.timing_class is not None:
            require_identifier(self.timing_class, "transfer.timing_class")
            require(
                self.kind == "timed", "transfer.timing_class", "requires timed work"
            )
        require(self.kind in {"copy", "timed"}, "transfer.kind", "unknown phase")
        require_non_negative(self.bytes, "transfer.bytes")
        require_non_negative(self.duration_ns, "transfer.duration_ns")
        require(
            isinstance(self.submission_ordered, bool),
            "transfer.submission_ordered",
            "expected bool",
        )
        if self.kind == "copy":
            require(self.resource is None, "transfer.resource", "copy uses its lane")
            require(self.duration_ns == 0, "transfer.duration_ns", "copy uses rates")
            require(
                not self.submission_ordered,
                "transfer.submission_ordered",
                "only timed phases are ordered",
            )
        else:
            require(
                self.resource is not None, "transfer.resource", "resource is required"
            )
            assert self.resource is not None
            require_identifier(self.resource, "transfer.resource")

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the phase as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "kind": self.kind,
            "bytes": self.bytes,
            "duration_ns": self.duration_ns,
            "resource": self.resource,
            "submission_ordered": self.submission_ordered,
            "timing_class": self.timing_class,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TransferPhase:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.alias_groups[0].transfer.fetch[0]"``.
            Error paths start with it.

        Returns
        -------
        TransferPhase
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        resource = field(data, "resource", path)
        return cls(
            expect_string(field(data, "kind", path), f"{path}.kind"),
            expect_integer(field(data, "bytes", path), f"{path}.bytes"),
            expect_integer(field(data, "duration_ns", path), f"{path}.duration_ns"),
            None if resource is None else expect_string(resource, f"{path}.resource"),
            expect_boolean(
                data.get("submission_ordered", False), f"{path}.submission_ordered"
            ),
            None
            if data.get("timing_class") is None
            else expect_string(data["timing_class"], f"{path}.timing_class"),
        )


@dataclass(frozen=True, slots=True)
class TransferSpec:
    """How an alias group moves when a move is more than one copy.

    The group is ready, or its source free, only after the whole recipe
    completes. The recipe prices a move for planning and simulation; the runtime
    performs the conversion itself.

    Parameters
    ----------
    spill_size_bytes : int
        Size of the group's spill copy, in bytes.
    fetch : tuple of TransferPhase
        The steps of a fetch, in order; at least one.
    evict : tuple of TransferPhase
        The steps of an eviction or write-back, in order; at least one.

    Raises
    ------
    ValidationError
        If ``spill_size_bytes`` is negative, a recipe is empty or not a tuple of
        phases, or a recipe has more than one submission-ordered phase.
    """

    spill_size_bytes: int
    fetch: tuple[TransferPhase, ...]
    evict: tuple[TransferPhase, ...]

    def __post_init__(self) -> None:
        require_non_negative(self.spill_size_bytes, "transfer.spill_size_bytes")
        for name in ("fetch", "evict"):
            phases = getattr(self, name)
            require_tuple(phases, f"transfer.{name}")
            require(bool(phases), f"transfer.{name}", "recipe must not be empty")
            require(
                all(isinstance(phase, TransferPhase) for phase in phases),
                f"transfer.{name}",
                "invalid phase",
            )
            require(
                sum(phase.submission_ordered for phase in phases) <= 1,
                f"transfer.{name}",
                "one ordered collective phase per transfer",
            )

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the recipe as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "spill_size_bytes": self.spill_size_bytes,
            "fetch": [phase.to_dict() for phase in self.fetch],
            "evict": [phase.to_dict() for phase in self.evict],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TransferSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.alias_groups[0].transfer"``.
            Error paths start with it.

        Returns
        -------
        TransferSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)

        def phases(name: str) -> tuple[TransferPhase, ...]:
            return tuple(
                TransferPhase.from_value(item, f"{path}.{name}[{index}]")
                for index, item in enumerate(
                    expect_list(field(data, name, path), f"{path}.{name}")
                )
            )

        return cls(
            expect_integer(
                field(data, "spill_size_bytes", path), f"{path}.spill_size_bytes"
            ),
            phases("fetch"),
            phases("evict"),
        )
