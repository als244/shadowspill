"""Deterministic allocator contract for one compiled structural task.

The task allocation ABI describes *what* the compiled callable asks from the
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

_TASK_ALLOCATION_ABI_SCHEMA = "shadowspill.task_allocation_abi/v2"


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
class TaskAllocationABIStep:
    """One operation in a pointer-free compiled-task allocator ABI.

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

    def __post_init__(self) -> None:
        if self.operation_index < 0 or self.allocation_ordinal < 0:
            raise ValueError("task allocation ABI ordinals must be non-negative")
        if not isinstance(self.operation, TaskAllocationOperation):
            raise TypeError("task allocation ABI operation is invalid")
        if self.requested_bytes < 0 or self.charged_bytes <= 0:
            raise ValueError("task allocation ABI sizes are invalid")
        if self.alignment_bytes <= 0:
            raise ValueError("task allocation ABI alignment must be positive")
        if any(value < 0 for value in self.output_leaf_indices):
            raise ValueError("task allocation ABI output leaves are invalid")
        if any(value < 0 for value in self.mutation_input_positions):
            raise ValueError("task allocation ABI mutation positions are invalid")
        if self.operation is TaskAllocationOperation.FREE and (
            self.output_leaf_indices
            or self.mutation_input_positions
            or self.persistent_after_task
        ):
            raise ValueError("task allocation ABI free carries allocation-only fields")

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
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationABIStep:
        if not isinstance(value, dict):
            raise ValueError("cached task allocation ABI step must be an object")
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
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("cached task allocation ABI step is invalid") from exc


@dataclass(frozen=True, slots=True)
class TaskAllocationABI:
    """Validated, deterministic allocator behavior for one structural task."""

    steps: tuple[TaskAllocationABIStep, ...]
    compatibility_digest: str

    def __post_init__(self) -> None:
        if len(self.compatibility_digest) != 64:
            raise ValueError("task allocation ABI digest must be SHA-256")
        if tuple(step.operation_index for step in self.steps) != tuple(
            range(len(self.steps))
        ):
            raise ValueError("task allocation ABI operation indices must be dense")
        _validate_steps(self.steps)
        if self.compatibility_digest != _digest_steps(self.steps):
            raise ValueError("task allocation ABI digest does not match its steps")

    @classmethod
    def capture(
        cls,
        trace: Sequence[TaskAllocationEvent],
        contract: TaskStorageContract | None = None,
    ) -> TaskAllocationABI:
        """Build an ABI from one normalized trace and offline mutation semantics."""

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
            _abi_step(index, event, mutation_by_leaf, freed_ordinals)
            for index, event in enumerate(trace)
        )
        return cls(steps, _digest_steps(steps))

    def for_retained_output_leaves(
        self,
        leaf_indices: Sequence[int],
    ) -> TaskAllocationABI:
        """Specialize profiled output destruction for one execution task.

        Isolated profiling destroys every returned tensor after inspecting its
        storage.  Repeated execution instead promotes only the leaves declared
        by the selected Program.  Retaining any view keeps its complete root
        allocation alive; terminal frees for that allocation are removed.
        Input-backed output leaves do not appear in this allocator ABI and are
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
        rewritten: list[TaskAllocationABIStep] = []
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
        return TaskAllocationABI(values, _digest_steps(values))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _TASK_ALLOCATION_ABI_SCHEMA,
            "compatibility_digest": self.compatibility_digest,
            "steps": [step.identity() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskAllocationABI:
        if not isinstance(value, dict):
            raise ValueError("cached task allocation ABI must be an object")
        if value.get("schema") != _TASK_ALLOCATION_ABI_SCHEMA:
            raise ValueError("cached task allocation ABI has an invalid schema")
        try:
            steps = tuple(
                TaskAllocationABIStep.from_dict(item) for item in value["steps"]
            )
            digest = str(value["compatibility_digest"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "cached task allocation ABI has an invalid schema"
            ) from exc
        return cls(steps, digest)


def _abi_step(
    operation_index: int,
    event: TaskAllocationEvent,
    mutation_by_leaf: dict[int, int],
    freed_ordinals: set[int],
) -> TaskAllocationABIStep:
    leaves = event.output_leaf_indices
    mutations = tuple(
        dict.fromkeys(
            mutation_by_leaf[leaf] for leaf in leaves if leaf in mutation_by_leaf
        )
    )
    return TaskAllocationABIStep(
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
    )


def _validate_steps(steps: tuple[TaskAllocationABIStep, ...]) -> None:
    live: dict[int, TaskAllocationABIStep] = {}
    retired: set[int] = set()
    returned_leaves: set[int] = set()
    next_allocation_ordinal = 0
    for step in steps:
        ordinal = step.allocation_ordinal
        if step.operation is TaskAllocationOperation.ALLOCATE:
            if ordinal != next_allocation_ordinal:
                raise ValueError(
                    "task allocation ABI requires dense allocation ordinals"
                )
            next_allocation_ordinal += 1
            if ordinal in live or ordinal in retired:
                raise ValueError("task allocation ABI allocates one ordinal twice")
            if returned_leaves.intersection(step.output_leaf_indices):
                raise ValueError("task allocation ABI returns one leaf twice")
            live[ordinal] = step
            returned_leaves.update(step.output_leaf_indices)
            continue
        allocated = live.pop(ordinal, None)
        if allocated is None:
            raise ValueError("task allocation ABI frees an unknown ordinal")
        if allocated.persistent_after_task:
            raise ValueError("task allocation ABI frees returned persistent storage")
        if (
            allocated.requested_bytes != step.requested_bytes
            or allocated.charged_bytes != step.charged_bytes
            or allocated.alignment_bytes != step.alignment_bytes
        ):
            raise ValueError("task allocation ABI changes geometry on free")
        retired.add(ordinal)
    if any(not step.persistent_after_task for step in live.values()):
        raise ValueError("task allocation ABI leaves anonymous storage live")


def _digest_steps(steps: tuple[TaskAllocationABIStep, ...]) -> str:
    encoded = json.dumps(
        {
            "schema": _TASK_ALLOCATION_ABI_SCHEMA,
            "steps": [step.identity() for step in steps],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = [
    "TaskAllocationABI",
    "TaskAllocationABIStep",
    "TaskAllocationEvent",
    "TaskAllocationOperation",
]
