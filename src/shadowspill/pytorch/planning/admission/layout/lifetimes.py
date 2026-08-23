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

from shadowspill.planner._lifetimes import LeaseLifetimes, build_lease_lifetimes
from shadowspill.planner._operations import AdmissionOperations
from shadowspill.simulator import SimulationResult

from ..admission_replay import AdmissionReplayPurpose
from ..setup import AdmissionSetup
from .model import FixedLayoutPlacement, LeaseLifetime

#: Compiled purpose codes, in the order `ShadowSpillAdmissionPurpose` declares.
_PURPOSES = (
    AdmissionReplayPurpose.INITIAL_OBJECT,
    AdmissionReplayPurpose.TASK_WORKSPACE,
    AdmissionReplayPurpose.TASK_OUTPUT,
    AdmissionReplayPurpose.MUTATION_REPLACEMENT,
    AdmissionReplayPurpose.RELEASE,
    AdmissionReplayPurpose.EVICTION,
    AdmissionReplayPurpose.FETCH_DESTINATION,
    AdmissionReplayPurpose.TERMINAL_COMPLETION,
)

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

    def placements(self, offsets: tuple[int, ...]) -> tuple[FixedLayoutPlacement, ...]:
        """Attach the chosen offsets to the fixed leases, named."""

        return tuple(
            self._placement(index, offsets[index])
            for index in range(self.leases.fixed_count)
        )

    @cached_property
    def initial_alias_leases(self) -> dict[str, int]:
        """Which lease each alias arrived resident in."""

        aliases = self.setup.alias_ids
        found: dict[str, int] = {}
        for index in range(self.leases.count):
            identity = self.leases.identities[index]
            if (
                _PURPOSES[identity.purpose] is AdmissionReplayPurpose.INITIAL_OBJECT
                and identity.alias != _NO_INDEX
            ):
                found[aliases[identity.alias]] = identity.lease_id
        return found

    @cached_property
    def action_destination_leases(self) -> dict[int, int]:
        """Which lease each fetch lands in."""

        found: dict[int, int] = {}
        for index in range(self.leases.count):
            identity = self.leases.identities[index]
            if (
                _PURPOSES[identity.purpose]
                is AdmissionReplayPurpose.FETCH_DESTINATION
                and identity.action != _NO_INDEX
            ):
                found[identity.action] = identity.lease_id
        return found

    @cached_property
    def task_allocation_leases(self) -> dict[tuple[str, int], int]:
        """Which lease each task allocation step ended up using."""

        found: dict[tuple[str, int], int] = {}
        for offset, step in enumerate(self.setup.allocation_steps):
            lease = self.leases.allocation_step_leases[offset]
            if step.allocates and lease != _NO_LEASE:
                found[(step.task_id, step.ordinal)] = lease
        return found

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
            None
            if identity.task == _NO_INDEX
            else self.setup.task_ids[identity.task],
            None
            if identity.alias == _NO_INDEX
            else self.setup.alias_ids[identity.alias],
            None if identity.action == _NO_INDEX else identity.action,
        )

    def _placement(self, index: int, offset: int) -> FixedLayoutPlacement:
        lifetime = self.leases.lifetimes[index]
        identity = self.leases.identities[index]
        task_id, alias_group_id, action_index = self._names(index)
        return FixedLayoutPlacement(
            lease_id=identity.lease_id,
            offset=offset,
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
    try:
        leases = build_lease_lifetimes(
            operations,
            setup.indexed_facts,
            simulation,
            dynamic_aliases=dynamic,
        )
    except RuntimeError:
        # The only caller error the library cannot describe is a terminal alias
        # that never reached a final lease. Resolve without them to name it.
        layout = LeaseLayout(
            leases=build_lease_lifetimes(
                operations, setup.indexed_facts, simulation
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
