"""Deterministic allocator contract for one compiled structural task.

The task allocation contract describes *what* the compiled callable asks from the
PyTorch allocator.  It deliberately excludes allocation IDs, pointers, slab
offsets, and runtime timing so the same fixed-shape callable has one stable
identity across processes and placement policies.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from shadowspill.pytorch.capture.storage import TaskStorageContract

_TASK_ALLOCATION_CONTRACT_SCHEMA = "shadowspill.task_allocation_contract/v1"


class TaskAllocationOperation(StrEnum):
    """One allocator transition made by a compiled task."""

    ALLOCATE = "allocate"
    FREE = "free"


@dataclass(frozen=True, slots=True)
class TaskAllocationEvent:
    """Task-local physical observation normalized away from process IDs."""

    allocation_ordinal: int
    operation: TaskAllocationOperation
    requested_bytes: int
    charged_bytes: int
    output_leaf_indices: tuple[int, ...] = ()
    output_view_offsets: tuple[int, ...] = ()
    reuses_ordinal: int | None = None
    alignment_bytes: int = 256

    def __post_init__(self) -> None:
        if self.allocation_ordinal < 0:
            raise ValueError("task allocation ordinal must be non-negative")
        if not isinstance(self.operation, TaskAllocationOperation):
            raise TypeError("task allocation operation has an invalid type")
        if self.requested_bytes < 0 or self.charged_bytes <= 0:
            raise ValueError("task allocation sizes are invalid")
        if self.alignment_bytes <= 0:
            raise ValueError("task allocation alignment must be positive")
        if any(index < 0 for index in self.output_leaf_indices):
            raise ValueError("task output leaf indices must be non-negative")
        if len(set(self.output_leaf_indices)) != len(self.output_leaf_indices):
            raise ValueError("task output leaf indices must be unique")
        if len(self.output_view_offsets) != len(self.output_leaf_indices):
            raise ValueError("task output leaves and view offsets must align")
        if any(offset < 0 for offset in self.output_view_offsets):
            raise ValueError("task output view offsets must be non-negative")
        if self.operation is TaskAllocationOperation.FREE and (
            self.output_leaf_indices or self.output_view_offsets
        ):
            raise ValueError("task free operation cannot publish output leaves")
        if self.reuses_ordinal is not None:
            if self.operation is not TaskAllocationOperation.ALLOCATE:
                raise ValueError("only an allocation may reuse a retired extent")
            if self.reuses_ordinal < 0:
                raise ValueError("reused task allocation ordinal must be non-negative")
            if self.reuses_ordinal == self.allocation_ordinal:
                raise ValueError("task allocation cannot reuse itself")

    def to_dict(self) -> dict[str, object]:
        return {
            "alignment_bytes": self.alignment_bytes,
            "allocation_ordinal": self.allocation_ordinal,
            "operation": self.operation.value,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
            "output_leaf_indices": list(self.output_leaf_indices),
            "output_view_offsets": list(self.output_view_offsets),
            "reuses_ordinal": self.reuses_ordinal,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationEvent:
        if not isinstance(value, dict):
            raise ValueError("cached allocation event must be an object")
        try:
            return cls(
                allocation_ordinal=int(value["allocation_ordinal"]),
                operation=TaskAllocationOperation(str(value["operation"])),
                requested_bytes=int(value["requested_bytes"]),
                charged_bytes=int(value["charged_bytes"]),
                output_leaf_indices=tuple(
                    int(item) for item in value["output_leaf_indices"]
                ),
                output_view_offsets=tuple(
                    int(item) for item in value["output_view_offsets"]
                ),
                reuses_ordinal=(
                    None
                    if value["reuses_ordinal"] is None
                    else int(value["reuses_ordinal"])
                ),
                alignment_bytes=int(value.get("alignment_bytes", 256)),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("cached allocation event has an invalid schema") from exc


@dataclass(frozen=True, slots=True)
class TaskAllocationContractStep:
    """One operation in a pointer-free compiled-task allocator contract.

    A persistent allocation is either returned storage (identified by output
    leaves) or bounded provider-owned state retained beyond the task boundary.
    """

    operation_index: int
    allocation_ordinal: int
    operation: TaskAllocationOperation
    requested_bytes: int
    charged_bytes: int
    alignment_bytes: int
    output_leaf_indices: tuple[int, ...] = ()
    mutation_input_positions: tuple[int, ...] = ()
    persistent_after_task: bool = False
    required: bool = False

    def __post_init__(self) -> None:
        if self.operation_index < 0 or self.allocation_ordinal < 0:
            raise ValueError("task allocation contract ordinals must be non-negative")
        if not isinstance(self.operation, TaskAllocationOperation):
            raise TypeError("task allocation contract operation is invalid")
        if self.requested_bytes < 0 or self.charged_bytes <= 0:
            raise ValueError("task allocation contract sizes are invalid")
        if self.alignment_bytes <= 0:
            raise ValueError("task allocation contract alignment must be positive")
        if any(value < 0 for value in self.output_leaf_indices):
            raise ValueError("task allocation contract output leaves are invalid")
        if any(value < 0 for value in self.mutation_input_positions):
            raise ValueError("task allocation contract mutation positions are invalid")
        if self.operation is TaskAllocationOperation.FREE and (
            self.output_leaf_indices
            or self.mutation_input_positions
            or self.persistent_after_task
            or self.required
        ):
            raise ValueError(
                "task allocation contract free carries allocation-only fields"
            )
        if self.required and not (
            self.output_leaf_indices or self.mutation_input_positions
        ):
            raise ValueError(
                "required task allocation contract storage must publish an output "
                "or mutation"
            )

    def identity(self) -> dict[str, object]:
        return {
            "operation_index": self.operation_index,
            "allocation_ordinal": self.allocation_ordinal,
            "operation": self.operation.value,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
            "alignment_bytes": self.alignment_bytes,
            "output_leaf_indices": list(self.output_leaf_indices),
            "mutation_input_positions": list(self.mutation_input_positions),
            "persistent_after_task": self.persistent_after_task,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationContractStep:
        if not isinstance(value, dict):
            raise ValueError("cached task allocation contract step must be an object")
        try:
            return cls(
                operation_index=int(value["operation_index"]),
                allocation_ordinal=int(value["allocation_ordinal"]),
                operation=TaskAllocationOperation(str(value["operation"])),
                requested_bytes=int(value["requested_bytes"]),
                charged_bytes=int(value["charged_bytes"]),
                alignment_bytes=int(value["alignment_bytes"]),
                output_leaf_indices=tuple(
                    int(item) for item in value["output_leaf_indices"]
                ),
                mutation_input_positions=tuple(
                    int(item) for item in value["mutation_input_positions"]
                ),
                persistent_after_task=bool(value["persistent_after_task"]),
                required=bool(value["required"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("cached task allocation contract step is invalid") from exc


@dataclass(frozen=True, slots=True)
class TaskAllocationContract:
    """Validated, deterministic allocator behavior for one structural task."""

    steps: tuple[TaskAllocationContractStep, ...]
    compatibility_digest: str

    def __post_init__(self) -> None:
        if len(self.compatibility_digest) != 64:
            raise ValueError("task allocation contract digest must be SHA-256")
        if tuple(step.operation_index for step in self.steps) != tuple(
            range(len(self.steps))
        ):
            raise ValueError(
                "task allocation contract operation indices must be contiguous"
            )
        _validate_steps(self.steps)
        if self.compatibility_digest != _digest_steps(self.steps):
            raise ValueError("task allocation contract digest does not match its steps")

    @classmethod
    def capture(
        cls,
        trace: Sequence[TaskAllocationEvent],
        contract: TaskStorageContract | None = None,
    ) -> TaskAllocationContract:
        """Build a contract from one trace and offline mutation semantics."""

        mutation_by_leaf = (
            {}
            if contract is None
            else {
                mutation.replacement_output_leaf: mutation.input_position
                for mutation in contract.mutations
                if mutation.replacement_output_leaf is not None
            }
        )
        freed_ordinals = {
            event.allocation_ordinal
            for event in trace
            if event.operation is TaskAllocationOperation.FREE
        }
        steps = tuple(
            _contract_step(index, event, mutation_by_leaf, freed_ordinals)
            for index, event in enumerate(trace)
        )
        return cls(steps, _digest_steps(steps))

    def for_retained_output_leaves(
        self,
        leaf_indices: Sequence[int],
    ) -> TaskAllocationContract:
        """Specialize profiled output destruction for one execution task.

        Isolated profiling destroys every returned tensor after inspecting its
        storage.  Repeated execution instead promotes only the leaves declared
        by the selected Program.  Retaining any view keeps its complete root
        allocation alive; terminal frees for that allocation are removed.
        Input-backed output leaves do not appear in this allocator contract and are
        intentionally ignored here.
        """

        retained_leaves = set(leaf_indices)
        if any(leaf < 0 for leaf in retained_leaves):
            raise ValueError("retained task output leaves must be non-negative")
        retained_ordinals = {
            step.allocation_ordinal
            for step in self.steps
            if step.operation is TaskAllocationOperation.ALLOCATE
            and retained_leaves.intersection(step.output_leaf_indices)
        }
        rewritten: list[TaskAllocationContractStep] = []
        for step in self.steps:
            if (
                step.operation is TaskAllocationOperation.FREE
                and step.allocation_ordinal in retained_ordinals
            ):
                continue
            retained_output = (
                step.operation is TaskAllocationOperation.ALLOCATE
                and step.allocation_ordinal in retained_ordinals
            )
            rewritten.append(
                replace(
                    step,
                    operation_index=len(rewritten),
                    persistent_after_task=(
                        step.persistent_after_task or retained_output
                    ),
                )
            )
        values = tuple(rewritten)
        return TaskAllocationContract(values, _digest_steps(values))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _TASK_ALLOCATION_CONTRACT_SCHEMA,
            "compatibility_digest": self.compatibility_digest,
            "steps": [step.identity() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationContract:
        if not isinstance(value, dict):
            raise ValueError("cached task allocation contract must be an object")
        if value.get("schema") != _TASK_ALLOCATION_CONTRACT_SCHEMA:
            raise ValueError("cached task allocation contract has an invalid schema")
        try:
            steps = tuple(
                TaskAllocationContractStep.from_dict(item) for item in value["steps"]
            )
            digest = str(value["compatibility_digest"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "cached task allocation contract has an invalid schema"
            ) from exc
        return cls(steps, digest)


@dataclass(frozen=True, slots=True)
class TaskAllocationPathObservation:
    """One repeated representative-input allocation-path probe."""

    probe_index: int
    repetition: int
    compatibility_digest: str
    operation_count: int
    allocation_count: int
    scratch_allocation_count: int
    scratch_maximum_requested_bytes: int
    scratch_maximum_charged_bytes: int
    scratch_peak_requested_bytes: int
    scratch_peak_charged_bytes: int
    scratch_terminal_charged_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.probe_index,
            self.repetition,
            self.operation_count,
            self.allocation_count,
            self.scratch_allocation_count,
            self.scratch_maximum_requested_bytes,
            self.scratch_maximum_charged_bytes,
            self.scratch_peak_requested_bytes,
            self.scratch_peak_charged_bytes,
            self.scratch_terminal_charged_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("task allocation path fields must be non-negative")
        if len(self.compatibility_digest) != 64:
            raise ValueError("task allocation path digest must be SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_index": self.probe_index,
            "repetition": self.repetition,
            "compatibility_digest": self.compatibility_digest,
            "operation_count": self.operation_count,
            "allocation_count": self.allocation_count,
            "scratch_allocation_count": self.scratch_allocation_count,
            "scratch_maximum_requested_bytes": (self.scratch_maximum_requested_bytes),
            "scratch_maximum_charged_bytes": self.scratch_maximum_charged_bytes,
            "scratch_peak_requested_bytes": self.scratch_peak_requested_bytes,
            "scratch_peak_charged_bytes": self.scratch_peak_charged_bytes,
            "scratch_terminal_charged_bytes": self.scratch_terminal_charged_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationPathObservation:
        if not isinstance(value, dict):
            raise ValueError("task allocation path observation must be an object")
        try:
            return cls(
                probe_index=int(value["probe_index"]),
                repetition=int(value["repetition"]),
                compatibility_digest=str(value["compatibility_digest"]),
                operation_count=int(value["operation_count"]),
                allocation_count=int(value["allocation_count"]),
                scratch_allocation_count=int(value["scratch_allocation_count"]),
                scratch_maximum_requested_bytes=int(
                    value["scratch_maximum_requested_bytes"]
                ),
                scratch_maximum_charged_bytes=int(
                    value["scratch_maximum_charged_bytes"]
                ),
                scratch_peak_requested_bytes=int(value["scratch_peak_requested_bytes"]),
                scratch_peak_charged_bytes=int(value["scratch_peak_charged_bytes"]),
                scratch_terminal_charged_bytes=int(
                    value["scratch_terminal_charged_bytes"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "task allocation path observation has an invalid schema"
            ) from exc


def compare_allocation_path(
    reference: TaskAllocationContract,
    observed: TaskAllocationContract,
    *,
    probe_index: int,
    repetition: int,
    operation_alignment: Sequence[tuple[int, int]] | None = None,
) -> TaskAllocationPathObservation:
    """Reconcile one observed path against an optional fixed-core contract."""

    if operation_alignment is not None:
        return _compare_aligned_allocation_path(
            reference,
            observed,
            probe_index=probe_index,
            repetition=repetition,
            operation_alignment=operation_alignment,
        )

    core_states = [0] * _allocation_count(reference.steps)
    actual_to_core: dict[int, int | None] = {}
    scratch_live: dict[int, tuple[int, int]] = {}
    scratch_live_requested = 0
    scratch_live_charged = 0
    scratch_peak_requested = 0
    scratch_peak_charged = 0
    scratch_maximum_requested = 0
    scratch_maximum_charged = 0
    scratch_count = 0
    core_index = 0

    for actual in observed.steps:
        if actual.operation is TaskAllocationOperation.ALLOCATE:
            match, skipped = _find_core_allocation(
                reference.steps,
                core_states,
                start=core_index,
                actual=actual,
            )
            if match is not None:
                for ordinal in skipped:
                    core_states[ordinal] = 2
                core_states[match.allocation_ordinal] = 1
                actual_to_core[actual.allocation_ordinal] = match.allocation_ordinal
                core_index = match.operation_index + 1
                continue
            if actual.required:
                raise ValueError(
                    "framework-visible allocation is absent from the fixed core: "
                    f"probe={probe_index}, repetition={repetition}, "
                    f"operation={actual.operation_index}, "
                    f"requested={actual.requested_bytes}, "
                    f"charged={actual.charged_bytes}"
                )
            actual_to_core[actual.allocation_ordinal] = None
            scratch_live[actual.allocation_ordinal] = (
                actual.requested_bytes,
                actual.charged_bytes,
            )
            scratch_live_requested += actual.requested_bytes
            scratch_live_charged += actual.charged_bytes
            scratch_peak_requested = max(scratch_peak_requested, scratch_live_requested)
            scratch_peak_charged = max(scratch_peak_charged, scratch_live_charged)
            scratch_maximum_requested = max(
                scratch_maximum_requested, actual.requested_bytes
            )
            scratch_maximum_charged = max(scratch_maximum_charged, actual.charged_bytes)
            scratch_count += 1
            continue

        mapped = actual_to_core.get(actual.allocation_ordinal)
        if actual.allocation_ordinal not in actual_to_core:
            raise ValueError("observed allocation path frees an unknown allocation")
        if mapped is None:
            requested, charged = scratch_live.pop(actual.allocation_ordinal)
            scratch_live_requested -= requested
            scratch_live_charged -= charged
            continue
        core_index = _skip_omitted_core_frees(reference.steps, core_states, core_index)
        if core_index >= len(reference.steps):
            raise ValueError("observed allocation path has an extra core free")
        expected = reference.steps[core_index]
        if (
            expected.operation is not TaskAllocationOperation.FREE
            or expected.allocation_ordinal != mapped
            or not _same_geometry(expected, actual)
        ):
            raise ValueError(
                "observed allocation path changes fixed-core free ordering: "
                f"probe={probe_index}, repetition={repetition}, "
                f"operation={actual.operation_index}, "
                f"mapped_core_ordinal={mapped}, "
                f"expected_core_operation={expected.operation_index}, "
                f"expected_core_ordinal={expected.allocation_ordinal}, "
                f"actual_ordinal={actual.allocation_ordinal}, "
                f"expected_geometry=({expected.requested_bytes},"
                f"{expected.charged_bytes},{expected.alignment_bytes}), "
                f"actual_geometry=({actual.requested_bytes},"
                f"{actual.charged_bytes},{actual.alignment_bytes})"
            )
        core_index += 1

    core_index = _finish_optional_core(reference.steps, core_states, core_index)
    if core_index != len(reference.steps):
        expected = reference.steps[core_index]
        raise ValueError(
            "observed allocation path omits required fixed-core behavior: "
            f"probe={probe_index}, repetition={repetition}, "
            f"core_operation={expected.operation_index}"
        )
    return TaskAllocationPathObservation(
        probe_index=probe_index,
        repetition=repetition,
        compatibility_digest=observed.compatibility_digest,
        operation_count=len(observed.steps),
        allocation_count=_allocation_count(observed.steps),
        scratch_allocation_count=scratch_count,
        scratch_maximum_requested_bytes=scratch_maximum_requested,
        scratch_maximum_charged_bytes=scratch_maximum_charged,
        scratch_peak_requested_bytes=scratch_peak_requested,
        scratch_peak_charged_bytes=scratch_peak_charged,
        scratch_terminal_charged_bytes=scratch_live_charged,
    )


def _compare_aligned_allocation_path(
    reference: TaskAllocationContract,
    observed: TaskAllocationContract,
    *,
    probe_index: int,
    repetition: int,
    operation_alignment: Sequence[tuple[int, int]],
) -> TaskAllocationPathObservation:
    """Classify scratch and omissions from one proven full-path alignment."""

    reference_to_observed = dict(operation_alignment)
    observed_to_reference = {
        observed_index: reference_index
        for reference_index, observed_index in operation_alignment
    }
    if len(reference_to_observed) != len(operation_alignment) or len(
        observed_to_reference
    ) != len(operation_alignment):
        raise ValueError("allocation operation alignment is not one-to-one")

    reference_operations = {step.operation_index: step for step in reference.steps}
    observed_operations = {step.operation_index: step for step in observed.steps}
    previous_reference = -1
    previous_observed = -1
    for reference_index, observed_index in operation_alignment:
        if reference_index <= previous_reference or observed_index <= previous_observed:
            raise ValueError("allocation operation alignment is not monotonic")
        try:
            expected = reference_operations[reference_index]
            actual = observed_operations[observed_index]
        except KeyError as error:
            raise ValueError(
                "allocation operation alignment is out of range"
            ) from error
        if not _same_operation(expected, actual):
            raise ValueError(
                "allocation operation alignment changes operation identity"
            )
        previous_reference = reference_index
        previous_observed = observed_index

    reference_lifetimes = _operation_lifetimes(reference.steps)
    observed_lifetimes = _operation_lifetimes(observed.steps)
    for ordinal, (allocation_index, free_index) in reference_lifetimes.items():
        matched_allocation = reference_to_observed.get(allocation_index)
        matched_free = (
            None if free_index is None else reference_to_observed.get(free_index)
        )
        allocation = reference.steps[allocation_index]
        if matched_allocation is None:
            if allocation.required:
                raise ValueError(
                    "observed allocation path omits required fixed-core behavior: "
                    f"probe={probe_index}, repetition={repetition}, "
                    f"core_operation={allocation_index}"
                )
            if matched_free is not None:
                raise ValueError(
                    "fixed-core lifetime has only its free operation matched"
                )
            continue
        actual_allocation = observed.steps[matched_allocation]
        actual_lifetime = observed_lifetimes[actual_allocation.allocation_ordinal]
        if (free_index is None) != (actual_lifetime[1] is None):
            raise ValueError("matched allocation changes terminal ownership")
        if free_index is not None and (
            matched_free is None or matched_free != actual_lifetime[1]
        ):
            raise ValueError(
                "observed allocation path changes fixed-core lifetime ordering: "
                f"probe={probe_index}, repetition={repetition}, "
                f"core_ordinal={ordinal}, observed_ordinal="
                f"{actual_allocation.allocation_ordinal}"
            )

    scratch_ordinals: set[int] = set()
    for ordinal, (allocation_index, free_index) in observed_lifetimes.items():
        matched_allocation = observed_to_reference.get(allocation_index)
        matched_free = (
            None if free_index is None else observed_to_reference.get(free_index)
        )
        allocation = observed.steps[allocation_index]
        if matched_allocation is not None:
            if free_index is not None and matched_free is None:
                raise ValueError(
                    "matched core lifetime has an unmatched free operation"
                )
            continue
        if allocation.required:
            raise ValueError(
                "framework-visible allocation is absent from the fixed core: "
                f"probe={probe_index}, repetition={repetition}, "
                f"operation={allocation.operation_index}, "
                f"requested={allocation.requested_bytes}, "
                f"charged={allocation.charged_bytes}"
            )
        if matched_free is not None:
            raise ValueError("scratch lifetime has a core-matched free operation")
        scratch_ordinals.add(ordinal)

    scratch_live_requested = 0
    scratch_live_charged = 0
    scratch_peak_requested = 0
    scratch_peak_charged = 0
    scratch_maximum_requested = 0
    scratch_maximum_charged = 0
    for step in observed.steps:
        if step.allocation_ordinal not in scratch_ordinals:
            continue
        if step.operation is TaskAllocationOperation.ALLOCATE:
            scratch_live_requested += step.requested_bytes
            scratch_live_charged += step.charged_bytes
            scratch_peak_requested = max(scratch_peak_requested, scratch_live_requested)
            scratch_peak_charged = max(scratch_peak_charged, scratch_live_charged)
            scratch_maximum_requested = max(
                scratch_maximum_requested, step.requested_bytes
            )
            scratch_maximum_charged = max(scratch_maximum_charged, step.charged_bytes)
        else:
            scratch_live_requested -= step.requested_bytes
            scratch_live_charged -= step.charged_bytes

    return TaskAllocationPathObservation(
        probe_index=probe_index,
        repetition=repetition,
        compatibility_digest=observed.compatibility_digest,
        operation_count=len(observed.steps),
        allocation_count=_allocation_count(observed.steps),
        scratch_allocation_count=len(scratch_ordinals),
        scratch_maximum_requested_bytes=scratch_maximum_requested,
        scratch_maximum_charged_bytes=scratch_maximum_charged,
        scratch_peak_requested_bytes=scratch_peak_requested,
        scratch_peak_charged_bytes=scratch_peak_charged,
        scratch_terminal_charged_bytes=scratch_live_charged,
    )


def _operation_lifetimes(
    steps: Sequence[TaskAllocationContractStep],
) -> dict[int, tuple[int, int | None]]:
    allocations: dict[int, int] = {}
    frees: dict[int, int] = {}
    for step in steps:
        if step.operation is TaskAllocationOperation.ALLOCATE:
            allocations[step.allocation_ordinal] = step.operation_index
        else:
            frees[step.allocation_ordinal] = step.operation_index
    return {
        ordinal: (operation_index, frees.get(ordinal))
        for ordinal, operation_index in allocations.items()
    }


def _same_operation(
    left: TaskAllocationContractStep,
    right: TaskAllocationContractStep,
) -> bool:
    return (
        left.operation is right.operation
        and left.requested_bytes == right.requested_bytes
        and left.charged_bytes == right.charged_bytes
        and left.alignment_bytes == right.alignment_bytes
        and left.output_leaf_indices == right.output_leaf_indices
        and left.mutation_input_positions == right.mutation_input_positions
        and left.persistent_after_task == right.persistent_after_task
        and left.required == right.required
    )


def _allocation_count(steps: Sequence[TaskAllocationContractStep]) -> int:
    return sum(step.operation is TaskAllocationOperation.ALLOCATE for step in steps)


def _same_geometry(
    left: TaskAllocationContractStep,
    right: TaskAllocationContractStep,
) -> bool:
    return (
        left.operation is right.operation
        and left.requested_bytes == right.requested_bytes
        and left.charged_bytes == right.charged_bytes
        and left.alignment_bytes == right.alignment_bytes
        and left.required == right.required
    )


def _find_core_allocation(
    steps: Sequence[TaskAllocationContractStep],
    states: list[int],
    *,
    start: int,
    actual: TaskAllocationContractStep,
) -> tuple[TaskAllocationContractStep | None, tuple[int, ...]]:
    skipped: list[int] = []
    skipped_set: set[int] = set()
    scan = start
    while scan < len(steps):
        candidate = steps[scan]
        if candidate.operation is TaskAllocationOperation.ALLOCATE:
            if _same_geometry(candidate, actual):
                return candidate, tuple(skipped)
            if candidate.required:
                return None, ()
            skipped.append(candidate.allocation_ordinal)
            skipped_set.add(candidate.allocation_ordinal)
            scan += 1
            continue
        ordinal = candidate.allocation_ordinal
        if states[ordinal] == 2 or ordinal in skipped_set:
            scan += 1
            continue
        return None, ()
    return None, ()


def _skip_omitted_core_frees(
    steps: Sequence[TaskAllocationContractStep],
    states: list[int],
    start: int,
) -> int:
    index = start
    while index < len(steps):
        step = steps[index]
        if (
            step.operation is TaskAllocationOperation.FREE
            and states[step.allocation_ordinal] == 2
        ):
            index += 1
            continue
        break
    return index


def _finish_optional_core(
    steps: Sequence[TaskAllocationContractStep],
    states: list[int],
    start: int,
) -> int:
    index = start
    while index < len(steps):
        index = _skip_omitted_core_frees(steps, states, index)
        if index >= len(steps):
            return index
        step = steps[index]
        if step.operation is not TaskAllocationOperation.ALLOCATE or step.required:
            return index
        states[step.allocation_ordinal] = 2
        index += 1
    return index


def _contract_step(
    operation_index: int,
    event: TaskAllocationEvent,
    mutation_by_leaf: dict[int, int],
    freed_ordinals: set[int],
) -> TaskAllocationContractStep:
    leaves = event.output_leaf_indices
    mutations = tuple(
        dict.fromkeys(
            mutation_by_leaf[leaf] for leaf in leaves if leaf in mutation_by_leaf
        )
    )
    return TaskAllocationContractStep(
        operation_index=operation_index,
        allocation_ordinal=event.allocation_ordinal,
        operation=event.operation,
        requested_bytes=event.requested_bytes,
        charged_bytes=event.charged_bytes,
        alignment_bytes=event.alignment_bytes,
        output_leaf_indices=leaves,
        mutation_input_positions=mutations,
        persistent_after_task=(
            event.operation is TaskAllocationOperation.ALLOCATE
            and event.allocation_ordinal not in freed_ordinals
        ),
        required=(
            event.operation is TaskAllocationOperation.ALLOCATE
            and bool(leaves or mutations)
        ),
    )


def _validate_steps(steps: tuple[TaskAllocationContractStep, ...]) -> None:
    live: dict[int, TaskAllocationContractStep] = {}
    retired: set[int] = set()
    returned_leaves: set[int] = set()
    next_allocation_ordinal = 0
    for step in steps:
        ordinal = step.allocation_ordinal
        if step.operation is TaskAllocationOperation.ALLOCATE:
            if ordinal != next_allocation_ordinal:
                raise ValueError(
                    "task allocation contract requires contiguous allocation ordinals"
                )
            next_allocation_ordinal += 1
            if ordinal in live or ordinal in retired:
                raise ValueError("task allocation contract allocates one ordinal twice")
            if returned_leaves.intersection(step.output_leaf_indices):
                raise ValueError("task allocation contract returns one leaf twice")
            live[ordinal] = step
            returned_leaves.update(step.output_leaf_indices)
            continue
        allocated = live.pop(ordinal, None)
        if allocated is None:
            raise ValueError("task allocation contract frees an unknown ordinal")
        if allocated.persistent_after_task:
            raise ValueError(
                "task allocation contract frees returned persistent storage"
            )
        if (
            allocated.requested_bytes != step.requested_bytes
            or allocated.charged_bytes != step.charged_bytes
            or allocated.alignment_bytes != step.alignment_bytes
        ):
            raise ValueError("task allocation contract changes geometry on free")
        retired.add(ordinal)
    if any(not step.persistent_after_task for step in live.values()):
        raise ValueError("task allocation contract leaves anonymous storage live")


def _digest_steps(steps: tuple[TaskAllocationContractStep, ...]) -> str:
    encoded = json.dumps(
        {
            "schema": _TASK_ALLOCATION_CONTRACT_SCHEMA,
            "steps": [step.identity() for step in steps],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = [
    "TaskAllocationContract",
    "TaskAllocationContractStep",
    "TaskAllocationEvent",
    "TaskAllocationOperation",
    "TaskAllocationPathObservation",
    "compare_allocation_path",
]
