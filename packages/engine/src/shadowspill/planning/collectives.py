"""CPU descriptions of the explicit collectives issued by saved Plans.

Workers and preparation use the same issue-order comparison. Opaque model/provider
collectives are not listed; their resource groups still prevent independent runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts.tensors import registration_name
from shadowspill.ir import MemoryActionKind

if TYPE_CHECKING:
    from shadowspill.artifacts.plan import Plan

COMMUNICATION_PHASES = ("gradient_communication", "weight_communication")


def collective_sequence(plan: Plan) -> tuple[tuple[str, str, str], ...]:
    """Collective-bearing calls in the native runtime's host submission order.

    A task issues its explicit collectives before its boundary actions. Within
    that boundary, collective write-backs precede collective fetches, retaining
    each direction's FIFO (runtime/tasks/task_boundaries.c). Comparing transfers
    and tasks independently misses fetches moved across optimizer collectives.
    """

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
    actions = defaultdict(list)
    for action in plan.schedule.actions:
        if action.alias_group_id not in sharded or action.kind not in (
            MemoryActionKind.FETCH,
            MemoryActionKind.EVICT,
            MemoryActionKind.WRITE_BACK,
        ):
            continue
        transfer = transfers.get(action.alias_group_id)
        if transfer is None:
            continue
        direction = "fetch" if action.kind == MemoryActionKind.FETCH else "evict"
        if any(
            phase.kind == "timed" and phase.bytes
            for phase in getattr(transfer, direction)
        ):
            actions[action.trigger_task_id].append(action)
    for task in costed.selected_tasks(plan.selections):
        if task.phase in COMMUNICATION_PHASES:
            touched = sorted(
                name
                for ref in (*task.inputs, *task.outputs)
                for name in by_object.get(getattr(ref, "object_id", ref), ())
            )
            rows.append(("task", str(task.phase), ",".join(touched)))
        boundary = actions.get(task.task_id, ())
        for fetch in (False, True):
            for action in boundary:
                if (action.kind == MemoryActionKind.FETCH) == fetch:
                    name = ",".join(sorted(by_alias.get(action.alias_group_id, ())))
                    rows.append(("transfer", action.kind.value, name))
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
            what = {
                "fetch": "a fetch",
                "evict": "an eviction",
                "write_back": "a write-back",
            }.get(operation, "a " + operation.replace("_", " ") + " task")
            return f"{what} of {tensors}" if tensors else what

        mine = reference[index] if index < len(reference) else None
        theirs = other[index] if index < len(other) else None
        return (
            f"collective {index + 1} of the {name} Plan is {describe(mine)} on "
            f"worker 0 but {describe(theirs)} on worker {worker}"
        )
    return None
