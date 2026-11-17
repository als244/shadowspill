"""The collectives a Plan's invocation issues, in the order it issues them.

Ranks pair their collectives by issue order, so ranks running different Plans
must issue the same collectives in the same order. The comparison interleaves
compound weight/gradient transfers with explicit communication tasks at their
actual task boundaries, including write-back-before-fetch ordering. Entries
name tensors, not the
identities a build assigns, so Plans built separately compare. Collectives a
model issues inside its own tasks, through communicator groups its provider
resources hold, are not listed; a Plan with such groups is never independent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import digest
from shadowspill.planning.collectives import collective_sequence, first_difference
from shadowspill.resources.description import resource_groups

if TYPE_CHECKING:
    from shadowspill.artifacts.plan import Plan
    from shadowspill.communication.binding.groups import GroupBindings


def independent_plans(paths: Mapping[str, Any]) -> frozenset[str]:
    """The named Plans that issue no collectives, neither in their transfers
    and communication tasks nor through communicator groups their model's
    resources hold: nothing such a Plan does has to meet another worker's
    work, so workers invoke it without meeting each other, and as many times
    as each needs."""

    return frozenset(
        name
        for name, path in paths.items()
        if not collective_sequence(path.plan)
        and not resource_groups(path.plan.program.resources)
    )


def check_collective_order(groups: GroupBindings, plans: Mapping[str, Plan]) -> None:
    """Refuse workers whose Plans would issue their collectives in different
    orders, naming the first difference, before any collective runs."""

    sequences = {
        name.removeprefix("named."): collective_sequence(plan)
        for name, plan in sorted(plans.items())
    }
    digests = groups.gather(
        digest({k: [list(r) for r in v] for k, v in sequences.items()})
    )
    if all(value == digests[0] for value in digests):
        return
    everyone = groups.gather(sequences)
    names = [sorted(item) for item in everyone]
    if any(item != names[0] for item in names):
        raise ValueError(f"workers hold different Plans: {names}")
    found = "their sequences differ"
    for name in sequences:
        difference = first_difference(name, [item[name] for item in everyone])
        if difference is not None:
            found = difference
            break
    raise ValueError(
        "workers would issue their collectives in different orders, and ranks "
        f"pair collectives by that order: {found}. Plans built for different "
        "GPUs fetch sharded weights in their own order; such ranks need "
        "replicated weight and gradient storage"
    )
