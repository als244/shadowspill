"""Deterministic finite-call packing with bounded repair and no extra buffers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from shadowspill.parallelism import InputPlacement
from shadowspill_client import BatchConfig
from shadowspill_client.protocol.inputs import InputSchema

from .shapes import PackingSchedule, packed_paths
from .validation import ValidatedBatch


class PackingError(ValueError):
    """The exact call has no packing found within its prepared bounded family."""


@dataclass(frozen=True)
class PackedAssignment:
    batch: ValidatedBatch
    schedule: PackingSchedule
    records: tuple[tuple[int, ...], ...]
    groups: tuple[tuple[tuple[int, ...], ...], ...]
    strategy: Literal["best_fit", "repair"]
    repair_nodes: int = 0


def pack_batch(
    batch: ValidatedBatch,
    schema: InputSchema,
    limits: BatchConfig,
    placement: InputPlacement,
    schedule: PackingSchedule,
    *,
    groups: Sequence[Sequence[int]] | None = None,
    max_repair_nodes: int = 10000,
) -> PackedAssignment:
    """Pack atomic Datum/groups; never split context or cross a call boundary.

    The declaration author must permit grouping/reordering and masked padding.
    Preparation separately proves collective order, memory and advertised input
    coverage. A bounded heuristic failure is explicit, never an extra buffer.
    """
    if batch.owner.closed:
        raise RuntimeError("input snapshot is closed")
    if type(max_repair_nodes) is not int or max_repair_nodes < 0:
        raise ValueError("max_repair_nodes must be a nonnegative integer")
    schedule.validate(schema, limits, placement)
    atomic = (
        tuple((i,) for i in range(len(batch.fields)))
        if groups is None
        else tuple(tuple(group) for group in groups)
    )
    indices = [i for group in atomic for i in group]
    if (
        any(not group for group in atomic)
        or any(type(i) is not int for i in indices)
        or sorted(indices) != list(range(len(batch.fields)))
    ):
        raise ValueError("atomic groups must cover every Datum exactly once")
    definitions = {tuple(field.path): field for field in schema.fields}
    buffers = schedule.buffers
    paths = buffers[0].fields
    packed = packed_paths(definitions, paths)
    shared = tuple(path for path in paths if definitions[path].batching == "shared")
    for values in batch.fields:
        if paths.keys() - values.keys():
            raise ValueError("physical buffer requires an absent validated input field")
    capacities = [buffer.resources(definitions) for buffer in buffers]
    costs = []
    compatible = []
    for group in atomic:
        cost = [len(group)]
        for path in packed:
            axis = definitions[path].pack_axis
            assert axis is not None
            cost.append(sum(batch.fields[i][path].shape[axis] for i in group))
        costs.append(tuple(cost))
        shared_ok = all(
            batch.fields[group[0]][path].equivalent(batch.fields[i][path])
            for path in shared
            for i in group[1:]
        )
        fits = []
        for buffer in buffers:
            shape_ok = True
            for index in group:
                for path, physical in buffer.fields.items():
                    field, actual = definitions[path], batch.fields[index][path].shape
                    offset = field.batching == "stack"
                    for axis, length in enumerate(actual):
                        if axis == field.pack_axis:
                            continue
                        if length > physical[axis + offset] or (
                            field.batching == "shared" and length != physical[axis]
                        ):
                            shape_ok = False
            fits.append(shape_ok and shared_ok)
        compatible.append(fits)

    maxima = [max(values) for values in zip(*capacities, strict=True)]
    order = sorted(
        range(len(atomic)),
        key=lambda i: (
            -max(
                (
                    cost / max(cap, 1)
                    for cost, cap in zip(costs[i], maxima, strict=True)
                ),
                default=0,
            ),
            -sum(costs[i]),
            i,
        ),
    )
    used = [[0] * len(maxima) for _ in buffers]
    assigned: list[list[int]] = [[] for _ in buffers]

    def candidates(group: int) -> list[int]:
        result = []
        for index, capacity in enumerate(capacities):
            if not compatible[group][index] or any(
                a + b > cap
                for a, b, cap in zip(used[index], costs[group], capacity, strict=True)
            ):
                continue
            if assigned[index] and any(
                not batch.fields[atomic[assigned[index][0]][0]][path].equivalent(
                    batch.fields[atomic[group][0]][path]
                )
                for path in shared
            ):
                continue
            result.append(index)
        return sorted(
            result,
            key=lambda i: (
                sum(
                    (cap - a - b) / max(cap, 1)
                    for a, b, cap in zip(
                        used[i], costs[group], capacities[i], strict=True
                    )
                ),
                i,
            ),
        )

    def assign(group: int, index: int, sign: int) -> None:
        for axis, size in enumerate(costs[group]):
            used[index][axis] += sign * size
        if sign == 1:
            assigned[index].append(group)
        else:
            assert assigned[index].pop() == group

    greedy = True
    for group_index in order:
        choices = candidates(group_index)
        if not choices:
            greedy = False
            break
        assign(group_index, choices[0], 1)
    nodes = 0
    if not greedy:
        assigned = [[] for _ in buffers]
        used = [[0] * len(maxima) for _ in buffers]
        shape_keys = [(b.max_datums, tuple(sorted(b.fields.items()))) for b in buffers]

        def repair(position: int) -> bool:
            nonlocal nodes
            if position == len(order):
                return True
            if nodes >= max_repair_nodes:
                return False
            nodes += 1
            group = order[position]
            empty_shapes = set()
            for index in candidates(group):
                if not assigned[index]:
                    if shape_keys[index] in empty_shapes:
                        continue
                    empty_shapes.add(shape_keys[index])
                assign(group, index, 1)
                if repair(position + 1):
                    return True
                assign(group, index, -1)
            return False

        # Large calls rely on the certified family construction, not unbounded
        # exponential search or Python recursion proportional to token count.
        if len(atomic) > 64 or not repair(0):
            raise PackingError(
                f"{schedule.name}: cannot pack this exact batch into its "
                f"{len(buffers)} prepared buffers (repair nodes {nodes}); "
                "use another explicitly prepared shape or submit explicit "
                "accumulation parts"
            )
    result_groups = tuple(tuple(atomic[i] for i in group) for group in assigned)
    return PackedAssignment(
        batch,
        schedule,
        tuple(tuple(i for group in groups for i in group) for groups in result_groups),
        result_groups,
        "best_fit" if greedy else "repair",
        nodes,
    )


def select_schedule(
    batch: ValidatedBatch,
    schema: InputSchema,
    limits: BatchConfig,
    placement: InputPlacement,
    candidates: Sequence[PackingSchedule],
    *,
    groups: Sequence[Sequence[int]] | None = None,
    max_repair_nodes: int = 10000,
) -> PackedAssignment:
    """Try only admitted candidates, in preparation's predicted-cost order."""
    failures = []
    for candidate in candidates:
        try:
            return pack_batch(
                batch,
                schema,
                limits,
                placement,
                candidate,
                groups=groups,
                max_repair_nodes=max_repair_nodes,
            )
        except PackingError as error:
            failures.append(str(error))
    raise PackingError(
        "no prepared schedule covers the submitted batch: " + "; ".join(failures)
    )
