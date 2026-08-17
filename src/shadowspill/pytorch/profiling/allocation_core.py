"""Evidence-derived fixed core for one structural task allocation path."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .allocation_contract import (
    TaskAllocationContract,
    TaskAllocationContractStep,
    TaskAllocationOperation,
    TaskAllocationPathObservation,
    compare_allocation_path,
)


@dataclass(frozen=True, slots=True)
class AllocationPathProbe:
    """One complete seed/repetition allocation path supplied to core derivation."""

    probe_index: int
    repetition: int
    allocation_contract: TaskAllocationContract

    def __post_init__(self) -> None:
        if self.probe_index < 0 or self.repetition < 0:
            raise ValueError("allocation path probe coordinates must be non-negative")


@dataclass(frozen=True, slots=True)
class DerivedAllocationCore:
    """Deterministic medoid path and every probe's reconciliation evidence."""

    allocation_contract: TaskAllocationContract
    source_digest: str
    weighted_edit_distance: int
    reference_edit_distance: int
    observations: tuple[TaskAllocationPathObservation, ...]


class AmbiguousAllocationPathError(ValueError):
    """A complete observed path has multiple minimum-edit core mappings."""


def derive_core_allocation_path(
    warmed_reference: TaskAllocationContract,
    probes: Sequence[AllocationPathProbe],
    *,
    warmed_reference_repetitions: int = 3,
) -> DerivedAllocationCore:
    """Choose and reconcile one fixed core from complete observed paths.

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
    score, _length, digest, core = min(ranked)

    core_tokens = tokenized[digest]
    observations: list[TaskAllocationPathObservation] = []
    for probe in probes:
        observed_tokens = _operation_tokens(probe.allocation_contract)
        probe_distance = _myers_edit_distance(core_tokens, observed_tokens)
        alignment = _unique_minimum_alignment(
            core_tokens,
            observed_tokens,
            probe_distance,
        )
        if alignment is None:
            raise AmbiguousAllocationPathError(
                "allocation path has multiple minimum-edit core/scratch "
                "interpretations: "
                f"probe={probe.probe_index}, repetition={probe.repetition}, "
                f"edit_distance={probe_distance}"
            )
        observations.append(
            compare_allocation_path(
                core,
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
        core_tokens,
        reference_tokens,
        reference_distance,
    )
    if reference_alignment is None:
        raise AmbiguousAllocationPathError(
            "warmed allocation path has multiple minimum-edit core/scratch "
            f"interpretations: edit_distance={reference_distance}"
        )
    compare_allocation_path(
        core,
        warmed_reference,
        probe_index=0,
        repetition=warmed_reference_repetitions,
        operation_alignment=reference_alignment,
    )
    return DerivedAllocationCore(
        allocation_contract=core,
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


def _myers_edit_distance(
    left: Sequence[tuple[object, ...]],
    right: Sequence[tuple[object, ...]],
) -> int:
    """Return insertion/deletion distance in O((N + M)D) time."""

    if left == right:
        return 0
    prefix = 0
    shared = min(len(left), len(right))
    while prefix < shared and left[prefix] == right[prefix]:
        prefix += 1
    left_end = len(left)
    right_end = len(right)
    while (
        left_end > prefix
        and right_end > prefix
        and left[left_end - 1] == right[right_end - 1]
    ):
        left_end -= 1
        right_end -= 1
    left = left[prefix:left_end]
    right = right[prefix:right_end]
    left_count = len(left)
    right_count = len(right)
    if left_count == 0 or right_count == 0:
        return left_count + right_count
    # Profiling paths are normally identical or differ by one optional
    # allocation lifetime.  Recognize that overwhelmingly common case with
    # one linear scan instead of constructing a Myers frontier.
    if abs(left_count - right_count) == 1 and _is_single_insertion(
        left,
        right,
    ):
        return 1
    frontier: dict[int, int] = {1: 0}
    for edits in range(left_count + right_count + 1):
        next_frontier: dict[int, int] = {}
        for diagonal in range(-edits, edits + 1, 2):
            if diagonal == -edits or (
                diagonal != edits
                and frontier.get(diagonal - 1, -1) < frontier.get(diagonal + 1, -1)
            ):
                x = frontier.get(diagonal + 1, 0)
            else:
                x = frontier.get(diagonal - 1, 0) + 1
            y = x - diagonal
            while x < left_count and y < right_count and left[x] == right[y]:
                x += 1
                y += 1
            next_frontier[diagonal] = x
            if x >= left_count and y >= right_count:
                return edits
        frontier = next_frontier
    raise AssertionError("edit-distance frontier did not terminate")


def _is_single_insertion(
    left: Sequence[tuple[object, ...]],
    right: Sequence[tuple[object, ...]],
) -> bool:
    """Return whether the longer sequence inserts exactly one token."""

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    if len(longer) != len(shorter) + 1:
        return False
    mismatch = 0
    while mismatch < len(shorter) and shorter[mismatch] == longer[mismatch]:
        mismatch += 1
    return shorter[mismatch:] == longer[mismatch + 1 :]


def _unique_minimum_alignment(
    left: Sequence[tuple[object, ...]],
    right: Sequence[tuple[object, ...]],
    distance: int,
) -> tuple[tuple[int, int], ...] | None:
    """Return the unique minimum-edit token mapping, if one exists.

    Forward and backward banded edit costs identify every equal-token pair
    that can participate in a minimum insertion/deletion alignment. If one
    token can pair with two positions, the physical core interpretation is
    ambiguous even when either choice would happen to execute successfully.
    """

    if distance == 0:
        return tuple((index, index) for index in range(len(left)))
    left_count = len(left)
    right_count = len(right)
    cells = (left_count + right_count + 2) * (2 * distance + 1)
    if cells > 20_000_000:
        raise ValueError(
            "allocation paths diverge too far for bounded core alignment: "
            f"left={left_count}, right={right_count}, distance={distance}"
        )
    forward: list[dict[int, int]] = [dict() for _ in range(left_count + 1)]
    for j in range(min(right_count, distance) + 1):
        forward[0][j] = j
    for i in range(1, left_count + 1):
        row = forward[i]
        previous = forward[i - 1]
        for j in range(max(0, i - distance), min(right_count, i + distance) + 1):
            candidates: list[int] = []
            if j in previous:
                candidates.append(previous[j] + 1)
            if j > 0 and j - 1 in row:
                candidates.append(row[j - 1] + 1)
            if j > 0 and left[i - 1] == right[j - 1] and j - 1 in previous:
                candidates.append(previous[j - 1])
            if candidates:
                value = min(candidates)
                if value <= distance:
                    row[j] = value

    backward: list[dict[int, int]] = [dict() for _ in range(left_count + 1)]
    for j in range(max(0, right_count - distance), right_count + 1):
        backward[left_count][j] = right_count - j
    for i in range(left_count - 1, -1, -1):
        row = backward[i]
        following = backward[i + 1]
        remaining_left = left_count - i
        center = right_count - remaining_left
        for j in range(
            min(right_count, center + distance),
            max(0, center - distance) - 1,
            -1,
        ):
            candidates = []
            if j in following:
                candidates.append(following[j] + 1)
            if j < right_count and j + 1 in row:
                candidates.append(row[j + 1] + 1)
            if j < right_count and left[i] == right[j] and j + 1 in following:
                candidates.append(following[j + 1])
            if candidates:
                value = min(candidates)
                if value <= distance:
                    row[j] = value

    left_matches: dict[int, int] = {}
    right_matches: dict[int, int] = {}
    possible_pairs: list[tuple[int, int]] = []
    for i, token in enumerate(left):
        for j in range(max(0, i - distance), min(right_count, i + distance + 1)):
            if token != right[j]:
                continue
            prefix = forward[i].get(j)
            suffix = backward[i + 1].get(j + 1)
            if prefix is None or suffix is None or prefix + suffix != distance:
                continue
            prior_right = left_matches.setdefault(i, j)
            prior_left = right_matches.setdefault(j, i)
            if prior_right != j or prior_left != i:
                return None
            possible_pairs.append((i, j))
    expected_matches_numerator = left_count + right_count - distance
    if expected_matches_numerator < 0 or expected_matches_numerator % 2 != 0:
        return None
    expected_matches = expected_matches_numerator // 2
    if len(possible_pairs) != expected_matches:
        return None
    previous_right = -1
    for _left_index, right_index in possible_pairs:
        if right_index <= previous_right:
            return None
        previous_right = right_index
    return tuple(possible_pairs)


__all__ = [
    "AllocationPathProbe",
    "AmbiguousAllocationPathError",
    "DerivedAllocationCore",
    "derive_core_allocation_path",
]
