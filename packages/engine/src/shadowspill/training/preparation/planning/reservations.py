"""Spill each Plan leaves for the persistent state its run's other operations keep."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.artifacts.program import Program
from shadowspill.artifacts.state import StateId
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.ir import MemoryLocation

from .requests import PlanningRequest


@dataclass(frozen=True)
class _Persistent:
    """A request's persistent spill: shared state by identity, the bytes no
    other Plan can name, and which identities are an accumulation's gradients."""

    shared: Mapping[StateId, int]
    private: int
    gradients: frozenset[StateId]


def _persistent(program: Program, rank: int) -> _Persistent:
    groups = {group.alias_group_id: group for group in program.alias_groups}
    entries = (*program.initial_residency, *program.final_residency)
    held = {e.alias_group_id for e in entries if e.location is MemoryLocation.SPILL}
    held |= {alias for alias, group in groups.items() if group.retain_spill_copy}
    identities: dict[str, list[StateId]] = {}
    for binding in program_state(program, rank).bindings:
        if binding.alias_group_id is not None:
            identities.setdefault(binding.alias_group_id, []).append(binding.identity)
    shared: dict[StateId, int] = {}
    gradients = set()
    private = 0
    for alias in held:
        size = groups[alias].spill_size_bytes
        names = identities.get(alias)
        if not names:
            private += size
            continue
        # Tied state shares one alias and the same identities in every Plan.
        key = min(names)
        shared[key] = size
        if any(name.role == "gradient" for name in names):
            gradients.add(key)
    return _Persistent(shared, private, frozenset(gradients))


def spill_reservations(
    requests: Mapping[str, PlanningRequest],
    candidates: Mapping[str, Sequence[str]],
    descriptions: Mapping[str, Mapping[str, object]],
    programs: Mapping[str, Program],
) -> dict[str, int]:
    """The spill each request's Plan leaves for persistent state it does not hold.

    A run's Plans, one candidate for each operation and batch, run one at a
    time against one spill pool, and each persistent spill lease is held for
    the whole run. A Plan searched within the spill budget less this
    reservation therefore leaves room for the state the other Plans keep:
    the members of its own candidate, such as a backward's update, and every
    other operation and batch's candidates. Shared state counts once, by
    identity, and the bytes only one Plan can name count at the largest of
    each other operation and batch's candidates. An accumulation's gradients
    are never held while a complete step runs, so a step leaves no room for
    them.

    Parameters
    ----------
    requests : mapping of str to PlanningRequest
        The Plans to search, by ID.
    candidates : mapping of str to sequence of str
        Each candidate's request IDs.
    descriptions : mapping of str to mapping
        Each candidate's operation and batch, as ``"operation"`` and
        ``"batch"``.
    programs : mapping of str to Program
        Each request's instantiated Program, by request ID.

    Returns
    -------
    dict of str to int
        Each request's reservation, in bytes; the largest over the candidates
        it serves.
    """

    state = {key: _persistent(programs[key], requests[key].rank) for key in requests}
    # On each rank, each operation and batch's candidates, as their requests.
    needs: dict[int, dict[str, list[tuple[str, ...]]]] = {}
    for candidate, members in candidates.items():
        description = descriptions[candidate]
        requirement = json.dumps(
            [description["operation"], description["batch"]], sort_keys=True
        )
        by_rank: dict[int, list[str]] = {}
        for key in members:
            by_rank.setdefault(requests[key].rank, []).append(key)
        for rank, keys in by_rank.items():
            needs.setdefault(rank, {}).setdefault(requirement, []).append(tuple(keys))
    reserved = dict.fromkeys(requests, 0)
    for rank_needs in needs.values():
        for requirement, witnesses in rank_needs.items():
            others = [
                groups for need, groups in rank_needs.items() if need != requirement
            ]
            for witness in witnesses:
                for key in witness:
                    beside = [other for other in witness if other != key]
                    shared: dict[StateId, int] = {}
                    gradients: set[StateId] = set()
                    for other in (
                        *beside,
                        *(member for groups in others for g in groups for member in g),
                    ):
                        shared.update(state[other].shared)
                        gradients |= state[other].gradients
                    private = sum(state[other].private for other in beside) + sum(
                        max(sum(state[member].private for member in g) for g in groups)
                        for groups in others
                    )
                    held = state[key].shared
                    step = requests[key].use.kind == "step"
                    extra = sum(
                        size
                        for identity, size in shared.items()
                        if identity not in held and not (step and identity in gradients)
                    )
                    reserved[key] = max(reserved[key], extra + private)
    return reserved
