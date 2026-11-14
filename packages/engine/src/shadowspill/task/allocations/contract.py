"""Stable task allocation geometry, mutations and returned-storage ownership."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from shadowspill.schema import artifact_schema

from ..storage import TaskStorageContract
from .events import TaskAllocationEvent, TaskAllocationOperation

_TASK_ALLOCATION_CONTRACT_SCHEMA = artifact_schema("task_allocation_contract")


@dataclass(frozen=True, slots=True)
class TaskAllocationContractStep:
    """One operation in a pointer-free compiled-task allocator contract.

    A persistent allocation is either returned storage (identified by output
    leaves) or bounded persistent in-pool state retained beyond the task boundary.
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

        Isolated profiling destroys every returned value after inspecting its
        storage.  Repeated execution instead promotes only the leaves declared
        by the selected ShadowSpillProgram.  Retaining any view keeps its complete root
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
