"""Allocator transitions normalized away from process IDs and pointers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TaskAllocationOperation(StrEnum):
    """One allocator transition a compiled task makes.

    Attributes
    ----------
    ALLOCATE : str
        An allocation is made.
    FREE : str
        An allocation is freed.
    """

    ALLOCATE = "allocate"
    FREE = "free"


@dataclass(frozen=True, slots=True)
class TaskAllocationEvent:
    """One allocation or free a task made, as profiling observed it.

    Allocations are numbered by ordinal in the order the task makes them, so
    the record holds no process identity or address.

    Parameters
    ----------
    allocation_ordinal : int
        Which allocation, from zero.
    operation : TaskAllocationOperation
        Whether it is made or freed.
    requested_bytes : int
        Bytes the task asked for.
    charged_bytes : int
        Bytes the allocator charged for this allocation, not the capacity of
        the address it takes. Positive.
    output_leaf_indices : tuple of int, default ()
        Positions of the outputs whose storage the allocation is. Empty for a
        free.
    output_view_offsets : tuple of int, default ()
        Where each of those outputs starts in the allocation, in bytes.
    reuses_ordinal : int, optional
        An earlier, freed allocation whose address this one takes. Allocations
        of temporary storage along one such chain may differ in size; admission
        reserves the largest.
    alignment_bytes : int, default 256
        The allocation's alignment, in bytes. Positive.

    Raises
    ------
    ValueError
        If a size, ordinal, leaf or offset is invalid, the leaves repeat or do
        not pair with the offsets, a free publishes outputs, or a free or the
        allocation itself is named as reused.
    TypeError
        If ``operation`` is not a `TaskAllocationOperation`.
    """

    allocation_ordinal: int
    operation: TaskAllocationOperation
    requested_bytes: int
    charged_bytes: int
    output_leaf_indices: tuple[int, ...] = ()
    output_view_offsets: tuple[int, ...] = ()
    reuses_ordinal: int | None = None
    alignment_bytes: int = 256

    def __post_init__(self) -> None:
        if self.allocation_ordinal < 0:
            raise ValueError("task allocation ordinal must be non-negative")
        if not isinstance(self.operation, TaskAllocationOperation):
            raise TypeError("task allocation operation has an invalid type")
        if self.requested_bytes < 0 or self.charged_bytes <= 0:
            raise ValueError("task allocation sizes are invalid")
        if self.alignment_bytes <= 0:
            raise ValueError("task allocation alignment must be positive")
        if any(index < 0 for index in self.output_leaf_indices):
            raise ValueError("task output leaf indices must be non-negative")
        if len(set(self.output_leaf_indices)) != len(self.output_leaf_indices):
            raise ValueError("task output leaf indices must be unique")
        if len(self.output_view_offsets) != len(self.output_leaf_indices):
            raise ValueError("task output leaves and view offsets must align")
        if any(offset < 0 for offset in self.output_view_offsets):
            raise ValueError("task output view offsets must be non-negative")
        if self.operation is TaskAllocationOperation.FREE and (
            self.output_leaf_indices or self.output_view_offsets
        ):
            raise ValueError("task free operation cannot publish output leaves")
        if self.reuses_ordinal is not None:
            if self.operation is not TaskAllocationOperation.ALLOCATE:
                raise ValueError("only an allocation may reuse a retired extent")
            if self.reuses_ordinal < 0:
                raise ValueError("reused task allocation ordinal must be non-negative")
            if self.reuses_ordinal == self.allocation_ordinal:
                raise ValueError("task allocation cannot reuse itself")

    def to_dict(self) -> dict[str, object]:
        """Return the event as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "alignment_bytes": self.alignment_bytes,
            "allocation_ordinal": self.allocation_ordinal,
            "operation": self.operation.value,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
            "output_leaf_indices": list(self.output_leaf_indices),
            "output_view_offsets": list(self.output_view_offsets),
            "reuses_ordinal": self.reuses_ordinal,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationEvent:
        """Read an event from its `to_dict` form.

        Parameters
        ----------
        value : dict
            The decoded JSON value. ``alignment_bytes`` defaults to 256 when
            absent.

        Returns
        -------
        TaskAllocationEvent
            The event.

        Raises
        ------
        ValueError
            If ``value`` is not a dict, lacks a field, or holds an invalid one.
        """
        if not isinstance(value, dict):
            raise ValueError("cached allocation event must be an object")
        try:
            return cls(
                allocation_ordinal=int(value["allocation_ordinal"]),
                operation=TaskAllocationOperation(str(value["operation"])),
                requested_bytes=int(value["requested_bytes"]),
                charged_bytes=int(value["charged_bytes"]),
                output_leaf_indices=tuple(
                    int(item) for item in value["output_leaf_indices"]
                ),
                output_view_offsets=tuple(
                    int(item) for item in value["output_view_offsets"]
                ),
                reuses_ordinal=(
                    None
                    if value["reuses_ordinal"] is None
                    else int(value["reuses_ordinal"])
                ),
                alignment_bytes=int(value.get("alignment_bytes", 256)),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("cached allocation event has an invalid schema") from exc
