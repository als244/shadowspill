"""Names for the leases the planner reported.

The library resolves every lease to a lifetime and an identity, and hands both
back as arrays of indices. This is the Python face of that: it makes the call,
and gives back a view that decodes an index to a name only when something
actually asks for one.

That split is the point. A measurement asks how many bytes a schedule needs
and reads one number, so it never decodes anything. Only a certificate names
leases, and only the ones it keeps.

The rules the library follows - where an operation sits, why a lease exists,
and the transitions that emit no operation at all - are specified in
docs/architecture/admission-leases.md, and stated readably in
reference/python/admission/lifetimes.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import chain

import numpy as np

from shadowspill.planner.admission.lifetimes import (
    LeaseLifetimes,
    build_lease_lifetimes,
)
from shadowspill.planner.admission.operations import AdmissionOperations
from shadowspill.simulator import SimulationResult
from shadowspill.simulator.indexing import IntervalArrays, interval_arrays_from_result

from ..admission_replay import AdmissionReplayPurpose
from ..setup import AdmissionSetup
from .model import (
    PURPOSES,
    FixedLayoutPlacements,
    FixedLayoutTaskAllocations,
    LeaseLifetime,
    column,
)

#: Compiled purpose codes, in the order `ShadowSpillAdmissionPurpose` declares.
_PURPOSES = PURPOSES
_INITIAL_OBJECT = PURPOSES.index(AdmissionReplayPurpose.INITIAL_OBJECT)
_FETCH_DESTINATION = PURPOSES.index(AdmissionReplayPurpose.FETCH_DESTINATION)

#: Initial residency names neither a task nor an action, and its index carries
#: no meaning; action boundaries name an action and its triggering task.
_INITIAL_BOUNDARY = 0
_ACTION_BOUNDARIES = frozenset({3, 4})

_NO_INDEX = (1 << 32) - 1
_NO_LEASE = (1 << 64) - 1


# Not slotted: the cached views below need an instance dictionary, and
# exactly one of these exists per measurement.
@dataclass(frozen=True)
class LeaseLayout:
    """One schedule's leases, named on demand.

    `leases.lifetimes[:fixed_count]` is what placement runs on; everything
    below decodes indices for a caller that needs names.
    """

    leases: LeaseLifetimes
    setup: AdmissionSetup

    @property
    def fixed_count(self) -> int:
        """Leases in the reusable fixed slice, which are the array's prefix."""

        return self.leases.fixed_count

    @cached_property
    def dynamic_lifetimes(self) -> tuple[LeaseLifetime, ...]:
        """The caller-owned leases held out of the fixed slice."""

        return tuple(
            self._lifetime(index)
            for index in range(self.leases.fixed_count, self.leases.count)
        )

    @cached_property
    def lifetime_records(self) -> np.ndarray:
        """Every lease's lifetime record, viewed in place as a structured array."""

        return np.ctypeslib.as_array(self.leases.lifetimes)[: self.leases.count]

    @cached_property
    def identity_records(self) -> np.ndarray:
        """Every lease's identity record, viewed in place as a structured array."""

        return np.ctypeslib.as_array(self.leases.identities)[: self.leases.count]

    def placements(self, offsets: np.ndarray) -> FixedLayoutPlacements:
        """Attach the chosen offsets to the fixed leases, as columns.

        Index columns hold the library's indices, with its no-index marker
        read as `NONE`: the 32-bit marker is all ones, which is -1 signed.
        """

        fixed = self.leases.fixed_count
        lifetimes = self.lifetime_records[:fixed]
        identities = self.identity_records[:fixed]
        return FixedLayoutPlacements(
            lease_id=column(identities["lease_id"], np.uint64),
            offset=column(offsets[:fixed], np.uint64),
            bytes=column(lifetimes["bytes"], np.uint64),
            alignment=column(lifetimes["alignment"], np.uint64),
            predicted_start_ns=column(lifetimes["start_ns"], np.uint64),
            predicted_end_ns=column(lifetimes["end_ns"], np.uint64),
            causal_start=column(identities["causal_start"], np.uint64),
            causal_end=column(identities["causal_end"], np.uint64),
            purpose=column(identities["purpose"], np.uint8),
            task=column(identities["task"].view(np.int32), np.int32),
            alias=column(identities["alias"].view(np.int32), np.int32),
            action=column(identities["action"].view(np.int32), np.int32),
            task_ids=self.setup.task_ids,
            alias_ids=self.setup.alias_ids,
        )

    def _leases_with(self, purpose: int, field: str) -> list[tuple[int, int]]:
        """`(index named by field, lease id)` for every lease of a purpose
        whose `field` names something, in lease order."""

        identities = self.identity_records
        named = identities[field]
        chosen = (identities["purpose"] == purpose) & (named != _NO_INDEX)
        return list(
            zip(
                named[chosen].tolist(),
                identities["lease_id"][chosen].tolist(),
                strict=True,
            )
        )

    @cached_property
    def initial_alias_leases(self) -> dict[str, int]:
        """Which lease each alias arrived resident in."""

        aliases = self.setup.alias_ids
        return {
            aliases[alias]: lease
            for alias, lease in self._leases_with(_INITIAL_OBJECT, "alias")
        }

    @cached_property
    def action_destination_leases(self) -> dict[int, int]:
        """Which lease each fetch lands in."""

        return dict(self._leases_with(_FETCH_DESTINATION, "action"))

    @cached_property
    def task_allocations(self) -> FixedLayoutTaskAllocations:
        """Which lease each task allocation step ended up using, in step order.

        A step that does not allocate names no lease, so every step that names
        one is an allocating step.
        """

        ordinals_by_task = self.setup.allocation_ordinals
        counts = np.fromiter(
            (len(ordinals) for ordinals in ordinals_by_task),
            dtype=np.int64,
            count=len(ordinals_by_task),
        )
        total = int(counts.sum())
        tasks = np.repeat(np.arange(len(ordinals_by_task), dtype=np.int32), counts)
        ordinals = np.fromiter(
            chain.from_iterable(ordinals_by_task), dtype=np.uint32, count=total
        )
        leases = np.ctypeslib.as_array(self.leases.allocation_step_leases)[:total]
        named = leases != _NO_LEASE
        return FixedLayoutTaskAllocations(
            task=column(tasks[named], np.int32),
            allocation_ordinal=column(ordinals[named], np.uint32),
            lease_id=column(leases[named], np.uint64),
            task_ids=self.setup.task_ids,
        )

    @cached_property
    def active_aliases(self) -> dict[str, int]:
        """The lease each alias still holds when the step ends."""

        aliases = self.setup.alias_ids
        return {
            alias: self.leases.alias_leases[index]
            for index, alias in enumerate(aliases)
            if self.leases.alias_leases[index] != _NO_LEASE
        }

    def _names(self, index: int) -> tuple[str | None, str | None, int | None]:
        """The task, alias and action an identity's indices stand for."""

        identity = self.leases.identities[index]
        return (
            None if identity.task == _NO_INDEX else self.setup.task_ids[identity.task],
            None
            if identity.alias == _NO_INDEX
            else self.setup.alias_ids[identity.alias],
            None if identity.action == _NO_INDEX else identity.action,
        )

    def _lifetime(self, index: int) -> LeaseLifetime:
        lifetime = self.leases.lifetimes[index]
        identity = self.leases.identities[index]
        task_id, alias_group_id, action_index = self._names(index)
        return LeaseLifetime(
            lease_id=identity.lease_id,
            bytes=lifetime.bytes,
            alignment=lifetime.alignment,
            predicted_start_ns=lifetime.start_ns,
            predicted_end_ns=lifetime.end_ns,
            causal_start=identity.causal_start,
            causal_end=identity.causal_end,
            purpose=_PURPOSES[identity.purpose],
            task_id=task_id,
            alias_group_id=alias_group_id,
            action_index=action_index,
        )


def resolve_lease_lifetimes(
    operations: AdmissionOperations,
    setup: AdmissionSetup,
    simulation: SimulationResult,
    dynamic_alias_group_ids: frozenset[str] = frozenset(),
) -> LeaseLayout:
    """Resolve one schedule's leases, holding the named aliases out of the slice.

    An alias may be produced, evicted and fetched again before caller handoff.
    Only the lease it holds at the final boundary escapes the reusable fixed
    slice; historical generations remain ordinary fixed lifetimes.
    """

    indices = {alias: index for index, alias in enumerate(setup.alias_ids)}
    unknown = sorted(dynamic_alias_group_ids - indices.keys())
    if unknown:
        raise ValueError(f"dynamic terminal aliases are not in this program: {unknown}")
    dynamic = tuple(sorted(indices[alias] for alias in dynamic_alias_group_ids))
    # A result the simulator produced carries its arrays; one read back from a
    # store is re-encoded into the template's index space.
    intervals = simulation.interval_arrays
    if not isinstance(intervals, IntervalArrays):
        intervals = interval_arrays_from_result(setup.template, simulation)
    try:
        leases = build_lease_lifetimes(
            operations,
            setup.indexed_facts,
            simulation,
            intervals,
            dynamic_aliases=dynamic,
        )
    except RuntimeError:
        # The only caller error the library cannot describe is a terminal alias
        # that never reached a final lease. Resolve without them to name it.
        layout = LeaseLayout(
            leases=build_lease_lifetimes(
                operations, setup.indexed_facts, simulation, intervals
            ),
            setup=setup,
        )
        missing = sorted(dynamic_alias_group_ids - layout.active_aliases.keys())
        if missing:
            raise ValueError(
                f"dynamic terminal aliases lack final execution leases: {missing}"
            ) from None
        raise
    return LeaseLayout(leases=leases, setup=setup)


__all__ = ["LeaseLayout", "resolve_lease_lifetimes"]
