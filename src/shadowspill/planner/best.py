"""The best plan a search has actually placed, shared by whoever is searching.

PressureFit plans one resolved program. A Program has several, and a plan
that fits under any one of them is a real plan -- so it can bound the search
under every other. This is the object that carries that bound across the
boundary, and its scope is the caller's decision:

- one instance threaded from resolved program to resolved program gives an
  ordered search where each one starts from the last one's answer;
- one instance per resolved program gives independent searches that share
  nothing, which is what parallelism without cross-program pruning looks
  like;
- one instance shared by concurrent searches prunes hardest, because every
  candidate sees a placement the moment it happens -- at the cost of a
  result that can depend on which worker arrived first.

The bound is a *placed* plan rather than merely a fast one, because placing
a plan is what costs: a plan no better than one already placed cannot become
the answer, so it is never measured. That is the whole reason this object is
on the hot path, and why it holds a record rather than a number -- whatever
holds it at the end is the plan the search selected.

The record lives in the compiled planner, which does its own locking. This
class owns its lifetime and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from .capi import CBestPlacedRecord, planner_api


@dataclass(frozen=True, slots=True)
class PlacedPlan:
    """What the search knows about the best plan it placed."""

    makespan_ns: int
    #: The capacity this plan was built against, which is a property of the
    #: plan rather than of the search that produced it.
    object_capacity_bytes: int
    #: How much device capacity the plan gave back. Re-timing or re-measuring
    #: it at any other capacity produces a different plan's timeline.
    capacity_given_back_bytes: int
    selection_index: int
    schedule_digest: bytes


class BestPlaced:
    """A placed plan to beat, readable by a running search."""

    def __init__(self) -> None:
        handle = planner_api().shadowspill_best_placed_create()
        if not handle:
            raise MemoryError("could not allocate the shared best-placed record")
        self._handle: int | None = handle

    @property
    def handle(self) -> int:
        """The pointer the compiled search consults. Zero once closed."""

        return 0 if self._handle is None else self._handle

    def read(self) -> PlacedPlan | None:
        """The plan held, or None if nothing has been placed."""

        if self._handle is None:
            return None
        record = CBestPlacedRecord()
        planner_api().shadowspill_best_placed_read(self._handle, record)
        if record.makespan_ns == 0:
            return None
        return PlacedPlan(
            makespan_ns=int(record.makespan_ns),
            object_capacity_bytes=int(record.object_capacity_bytes),
            capacity_given_back_bytes=int(record.capacity_given_back_bytes),
            selection_index=int(record.selection_index),
            schedule_digest=bytes(record.schedule_digest),
        )

    def close(self) -> None:
        if self._handle is not None:
            planner_api().shadowspill_best_placed_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> BestPlaced:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["BestPlaced", "PlacedPlan"]
