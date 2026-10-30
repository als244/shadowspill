"""The collectives a Plan's invocation issues, in the order it issues them.

Ranks pair their collectives by issue order, so ranks running different Plans
must issue the same collectives in the same order. A Plan issues them on two
paths: the compound transfers that fetch sharded weights and evict sharded
gradients, in its schedule's order, and the gradient-communication and
weight-communication tasks, in its task order. Entries name tensors, not the
identities a build assigns, so Plans built separately compare.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.tensors import registration_name
from shadowspill.ir import MemoryActionKind

if TYPE_CHECKING:
    from shadowspill.artifacts.plan import Plan
    from shadowspill.communication.binding.groups import GroupBindings

COMMUNICATION_PHASES = ("gradient_communication", "weight_communication")


def collective_sequence(plan: Plan) -> tuple[tuple[str, str, str], ...]:
    """``(path, operation, tensors)`` for every collective, in issue order."""

    program = plan.program
    bindings = program.bindings
    alias_of = {item.object_id: item.alias_group_id for item in program.objects}
    by_object: dict[str, list[str]] = {}
    for item in bindings.get("registrations", ()):
        by_object.setdefault(item["object_id"], []).append(registration_name(item))
    for item in bindings.get("gradients", ()):
        if "gradient_object_id" in item:
            by_object.setdefault(item["gradient_object_id"], []).append(
                item["parameter_name"] + ".grad"
            )
    by_alias: dict[str, list[str]] = {}
    for object_id, found in by_object.items():
        by_alias.setdefault(alias_of.get(object_id, object_id), []).extend(found)
    sharded = set(bindings.get("weight_shards", {})) | set(
        bindings.get("gradient_shards", {})
    )
    costed = plan._costed_program
    transfers = {item.alias_group_id: item.transfer for item in costed.alias_groups}
    rows: list[tuple[str, str, str]] = []
    for action in plan.schedule.actions:
        if action.alias_group_id not in sharded or action.kind not in (
            MemoryActionKind.FETCH,
            MemoryActionKind.EVICT,
        ):
            continue
        transfer = transfers.get(action.alias_group_id)
        if transfer is None:
            continue
        moved = sum(
            phase.bytes
            for phase in getattr(transfer, action.kind.value)
            if phase.kind == "timed"
        )
        if moved:
            name = ",".join(sorted(by_alias.get(action.alias_group_id, ())))
            rows.append(("transfer", action.kind.value, name))
    for task in costed.selected_tasks(plan.selections):
        if task.phase in COMMUNICATION_PHASES:
            touched = sorted(
                name
                for ref in (*task.inputs, *task.outputs)
                for name in by_object.get(getattr(ref, "object_id", ref), ())
            )
            rows.append(("task", str(task.phase), ",".join(touched)))
    return tuple(rows)


def first_difference(
    name: str, sequences: list[tuple[tuple[str, str, str], ...]]
) -> str | None:
    """Where the workers' sequences for the Plan called ``name`` first part,
    in words, or ``None`` when every worker issues the same sequence."""

    reference = sequences[0]
    for worker, other in enumerate(sequences[1:], start=1):
        if other == reference:
            continue
        index = next(
            (
                i
                for i, (a, b) in enumerate(zip(reference, other, strict=False))
                if a != b
            ),
            min(len(reference), len(other)),
        )

        def describe(entry: Any) -> str:
            if entry is None:
                return "no collective"
            _, operation, tensors = entry
            what = {"fetch": "a fetch", "evict": "an eviction"}.get(
                operation, "a " + operation.replace("_", " ") + " task"
            )
            return f"{what} of {tensors}" if tensors else what

        mine = reference[index] if index < len(reference) else None
        theirs = other[index] if index < len(other) else None
        return (
            f"collective {index + 1} of the {name} Plan is {describe(mine)} on "
            f"worker 0 but {describe(theirs)} on worker {worker}"
        )
    return None


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
