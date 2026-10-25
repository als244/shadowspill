"""Derive gradient SUMs from logical copies and the destination's ownership."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .ownership import ParameterOwnership


@dataclass(frozen=True)
class ReductionPhase:
    operation: Literal["reduce_scatter", "all_reduce"]
    groups: tuple[tuple[int, ...], ...]
    input_elements: int
    output_elements: int


@dataclass(frozen=True)
class GradientReduction:
    """SUM contributions once across logical copies into the declared ownership.

    Restored accumulators inject each previously reduced element on one rank;
    reducing again then preserves that total while adding new contributions.
    """

    parameter_name: str
    source_elements: int
    padded_elements: int
    result_elements: int
    phases: tuple[ReductionPhase, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gradient_reduction(
    parameter: ParameterOwnership,
    *,
    destination: Literal["gradients", "optimizer"] = "gradients",
) -> GradientReduction:
    """SUM once across every copy, leaving equal padded owned ranges.

    First reduce-scatter within each storage group. Then all-reduce matching
    ranges across groups that hold the same logical parameter subset. The
    second phase is still required for repeated complete-model replicas and
    repeated model-shard groups. Objective normalization belongs to the caller.
    """
    if not parameter.trainable:
        raise ValueError(f"{parameter.name}: frozen parameters have no gradient SUM")
    storage = getattr(parameter, destination)
    size = len(storage.groups[0])
    capacity = storage.capacity
    padded = capacity * size
    phases = []
    if size > 1:
        phases.append(
            ReductionPhase("reduce_scatter", storage.groups, padded, capacity)
        )
    corresponding: list[tuple[int, ...]] = []
    for copies in parameter.copy_groups:
        groups = [group for group in storage.groups if group[0] in copies]
        if sorted(rank for group in groups for rank in group) != sorted(copies):
            raise ValueError(
                "gradient storage groups must partition logical parameter copies"
            )
        corresponding.extend(
            tuple(group[index] for group in groups) for index in range(size)
        )
    if len(corresponding[0]) > 1:
        phases.append(
            ReductionPhase("all_reduce", tuple(corresponding), capacity, capacity)
        )
    return GradientReduction(
        parameter.name, storage.elements, padded, capacity, tuple(phases)
    )
