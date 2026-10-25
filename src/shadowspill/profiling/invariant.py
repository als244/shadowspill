"""The invariant part of one structural task allocation path, from evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationContractStep,
    TaskAllocationOperation,
    TaskAllocationPathObservation,
    compare_allocation_path,
)

from .alignment import _myers_edit_distance, _unique_minimum_alignment


@dataclass(frozen=True, slots=True)
class AllocationPathProbe:
    """One complete seed/repetition allocation path supplied to invariant derivation."""

    probe_index: int
    repetition: int
    allocation_contract: TaskAllocationContract

    def __post_init__(self) -> None:
        if self.probe_index < 0 or self.repetition < 0:
            raise ValueError("allocation path probe coordinates must be non-negative")


@dataclass(frozen=True, slots=True)
class DerivedAllocationInvariant:
    """Deterministic medoid path and every probe's reconciliation evidence."""

    allocation_contract: TaskAllocationContract
    source_digest: str
    weighted_edit_distance: int
    reference_edit_distance: int
    observations: tuple[TaskAllocationPathObservation, ...]


class AmbiguousAllocationPathError(ValueError):
    """A complete observed path has multiple minimum-edit invariant mappings."""


def derive_invariant_allocation_path(
    warmed_reference: TaskAllocationContract,
    probes: Sequence[AllocationPathProbe],
    *,
    warmed_reference_repetitions: int = 3,
) -> DerivedAllocationInvariant:
    """Choose and reconcile one invariant from complete observed paths.

    The medoid minimizes total insertion/deletion distance across the warmed
    reference and the seed/repetition matrix. Repeated identical paths carry
    their observed multiplicity. A shorter path and then its digest break an
    otherwise exact tie, which keeps one-time insertions in dynamic scratch.

    Reconciliation remains lifetime-aware through ``compare_allocation_path``:
    an allocation match is valid only when its paired free and semantic
    ownership remain ordered consistently.
    """

    if warmed_reference_repetitions < 1:
        raise ValueError("warmed reference repetition count must be positive")
    samples = (warmed_reference, *(probe.allocation_contract for probe in probes))
    weights = Counter(item.compatibility_digest for item in samples)
    weights[warmed_reference.compatibility_digest] += warmed_reference_repetitions - 1
    candidates = {item.compatibility_digest: item for item in samples}
    tokenized = {digest: _operation_tokens(item) for digest, item in candidates.items()}
    distances: dict[tuple[str, str], int] = {}

    def distance(left: str, right: str) -> int:
        key = (left, right) if left <= right else (right, left)
        if key not in distances:
            distances[key] = _myers_edit_distance(tokenized[key[0]], tokenized[key[1]])
        return distances[key]

    ranked: list[tuple[int, int, str, TaskAllocationContract]] = []
    for digest, candidate in candidates.items():
        score = sum(
            count * distance(digest, observed_digest)
            for observed_digest, count in weights.items()
        )
        ranked.append((score, len(candidate.steps), digest, candidate))
    score, _length, digest, invariant = min(ranked)

    invariant_tokens = tokenized[digest]
    observations: list[TaskAllocationPathObservation] = []
    for probe in probes:
        observed_tokens = _operation_tokens(probe.allocation_contract)
        probe_distance = _myers_edit_distance(invariant_tokens, observed_tokens)
        alignment = _unique_minimum_alignment(
            invariant_tokens,
            observed_tokens,
            probe_distance,
        )
        if alignment is None:
            raise AmbiguousAllocationPathError(
                "allocation path has multiple minimum-edit invariant/scratch "
                "interpretations: "
                f"probe={probe.probe_index}, repetition={probe.repetition}, "
                f"edit_distance={probe_distance}"
            )
        observations.append(
            compare_allocation_path(
                invariant,
                probe.allocation_contract,
                probe_index=probe.probe_index,
                repetition=probe.repetition,
                operation_alignment=alignment,
            )
        )
    # The warmed reference is also required to reconcile if another observed
    # path wins the medoid. This guards the physical baseline used for timing.
    reference_tokens = _operation_tokens(warmed_reference)
    reference_distance = distance(digest, warmed_reference.compatibility_digest)
    reference_alignment = _unique_minimum_alignment(
        invariant_tokens,
        reference_tokens,
        reference_distance,
    )
    if reference_alignment is None:
        raise AmbiguousAllocationPathError(
            "warmed allocation path has multiple minimum-edit invariant/scratch "
            f"interpretations: edit_distance={reference_distance}"
        )
    compare_allocation_path(
        invariant,
        warmed_reference,
        probe_index=0,
        repetition=warmed_reference_repetitions,
        operation_alignment=reference_alignment,
    )
    return DerivedAllocationInvariant(
        allocation_contract=invariant,
        source_digest=digest,
        weighted_edit_distance=score,
        reference_edit_distance=reference_distance,
        observations=tuple(observations),
    )


def _operation_tokens(
    contract: TaskAllocationContract,
) -> tuple[tuple[object, ...], ...]:
    allocations = {
        step.allocation_ordinal: _allocation_token(step)
        for step in contract.steps
        if step.operation is TaskAllocationOperation.ALLOCATE
    }
    return tuple(
        _allocation_token(step)
        if step.operation is TaskAllocationOperation.ALLOCATE
        else ("free", *allocations[step.allocation_ordinal])
        for step in contract.steps
    )


def _allocation_token(step: TaskAllocationContractStep) -> tuple[object, ...]:
    return (
        "allocate",
        step.requested_bytes,
        step.charged_bytes,
        step.alignment_bytes,
        step.output_leaf_indices,
        step.mutation_input_positions,
        step.persistent_after_task,
        step.required,
    )


__all__ = [
    "AllocationPathProbe",
    "AmbiguousAllocationPathError",
    "DerivedAllocationInvariant",
    "derive_invariant_allocation_path",
]
