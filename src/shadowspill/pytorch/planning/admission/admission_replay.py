"""Cross-task physical admission through the production ``MemoryPool``.

This module deliberately operates at the same abstraction as the simulator:
one task-workspace envelope, persistent object generations, and ordered memory
actions.  It never consumes task-local allocator callback traces.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    Program,
    RecomputationSelection,
    TaskSpec,
)
from shadowspill.planner import AdmissionTopology, StorageHandoff
from shadowspill.runtime import (
    AdmissionReplayOperation as PoolAdmissionOperation,
)
from shadowspill.runtime import (
    AdmissionReplayOperationKind,
    AdmissionReplayResult,
    AdmissionReuseDependency,
    run_admission_replay,
)

from .bindings import TaskOutputBinding, build_admission_topology


class AdmissionReplayPurpose(StrEnum):
    """Semantic reason for one production-pool transition."""

    INITIAL_OBJECT = "initial_object"
    TASK_WORKSPACE = "task_workspace"
    TASK_OUTPUT = "task_output"
    MUTATION_REPLACEMENT = "mutation_replacement"
    RELEASE = "release"
    EVICTION = "eviction"
    FETCH_DESTINATION = "fetch_destination"
    TERMINAL_COMPLETION = "terminal_completion"


class OwnershipTransitionKind(StrEnum):
    """Object ownership transition that requires no pool operation."""

    STORAGE_HANDOFF = "storage_handoff"
    MUTATION_REPLACEMENT = "mutation_replacement"


@dataclass(frozen=True, slots=True)
class AdmissionReplayStep:
    """One low-level pool operation with its task-level provenance."""

    operation: PoolAdmissionOperation
    purpose: AdmissionReplayPurpose
    task_id: str | None = None
    alias_group_id: str | None = None
    action_index: int | None = None


@dataclass(frozen=True, slots=True)
class OwnershipTransition:
    """One alias-generation change that preserves or replaces a lease."""

    task_id: str
    kind: OwnershipTransitionKind
    destination_alias_group_id: str
    destination_lease_id: int
    source_alias_group_id: str | None = None
    source_lease_id: int | None = None


@dataclass(frozen=True, slots=True)
class CausalAdmissionDependency:
    """Physical reuse edge emitted by the production allocation policy."""

    dependency_id: int
    predecessor_lease_id: int
    predecessor_task_id: str
    predecessor_alias_group_id: str
    predecessor_action_index: int
    successor_lease_id: int
    successor_task_id: str | None
    successor_alias_group_id: str | None
    successor_action_index: int | None
    consumer_operation_index: int


@dataclass(frozen=True, slots=True)
class AdmissionReplay:
    """Timing-free physical certificate for one selected execution schedule."""

    pool: AdmissionReplayResult
    operations: tuple[AdmissionReplayStep, ...]
    ownership_transitions: tuple[OwnershipTransition, ...]
    dependencies: tuple[CausalAdmissionDependency, ...]
    final_execution_aliases: tuple[str, ...]
    workspace_bytes_by_task: tuple[tuple[str, int], ...]
    compatibility_digest: str


@dataclass(frozen=True, slots=True)
class _LeaseProvenance:
    purpose: AdmissionReplayPurpose
    task_id: str | None = None
    alias_group_id: str | None = None
    action_index: int | None = None


@dataclass(frozen=True, slots=True)
class _PendingEviction:
    lease_id: int
    dependency_id: int
    task_id: str
    alias_group_id: str
    action_index: int


class _AdmissionScriptBuilder:
    """Build one deterministic task-boundary script for ``MemoryPool``."""

    def __init__(
        self,
        program: Program,
        schedule: MemorySchedule,
        selections: tuple[RecomputationSelection, ...],
        topology: AdmissionTopology,
    ) -> None:
        self.program = program
        self.schedule = schedule
        self.tasks = program.selected_tasks(selections)
        self.admission_by_task = {item.task_id: item for item in topology.tasks}
        self.alignment = topology.minimum_alignment
        self.alias_size = {
            alias.alias_group_id: alias.size_bytes for alias in program.alias_groups
        }
        self.alias_by_object = {
            item.object_id: item.alias_group_id for item in program.objects
        }
        self.actions_by_task = self._index_actions()
        self.operations: list[AdmissionReplayStep] = []
        self.ownership_transitions: list[OwnershipTransition] = []
        self.active_aliases: dict[str, int] = {}
        self.lease_provenance: dict[int, _LeaseProvenance] = {}
        self.pending_evictions: list[_PendingEviction] = []
        self.handoff_releases: set[tuple[str, str]] = set()
        self.workspace_bytes_by_task: list[tuple[str, int]] = []
        self.next_lease_id = 0
        self.next_dependency_id = 0

    def build(
        self,
    ) -> tuple[
        tuple[AdmissionReplayStep, ...],
        tuple[OwnershipTransition, ...],
        tuple[_PendingEviction, ...],
        tuple[str, ...],
        tuple[tuple[str, int], ...],
        int,
        int,
    ]:
        self._acquire_initial_objects()
        for task in self.tasks:
            self._apply_task(task)
        self._complete_pending_evictions()
        self._validate_final_execution_residency()
        return (
            tuple(self.operations),
            tuple(self.ownership_transitions),
            tuple(self.pending_evictions),
            tuple(sorted(self.active_aliases)),
            tuple(self.workspace_bytes_by_task),
            self.next_lease_id,
            self.next_dependency_id,
        )

    def _index_actions(self) -> dict[str, tuple[tuple[int, MemoryAction], ...]]:
        indexed: dict[str, list[tuple[int, MemoryAction]]] = {}
        for index, action in enumerate(self.schedule.actions):
            indexed.setdefault(action.trigger_task_id, []).append((index, action))
        return {task_id: tuple(actions) for task_id, actions in indexed.items()}

    def _acquire_initial_objects(self) -> None:
        for residency in self.schedule.initial_residency:
            alias_id = residency.alias_group_id
            if (
                residency.location is not MemoryLocation.DEVICE
                or self.alias_size[alias_id] == 0
            ):
                continue
            self.active_aliases[alias_id] = self._acquire(
                self.alias_size[alias_id],
                _LeaseProvenance(
                    AdmissionReplayPurpose.INITIAL_OBJECT,
                    alias_group_id=alias_id,
                ),
            )

    def _apply_task(self, task: TaskSpec) -> None:
        self._validate_task_inputs(task)
        admission = self.admission_by_task[task.task_id]
        handoffs = admission.storage_handoffs
        replacements = admission.replacement_aliases
        replacement_aliases = set(replacements)
        workspace_bytes = admission.workspace_bytes
        self.workspace_bytes_by_task.append((task.task_id, workspace_bytes))
        workspace_lease = (
            None
            if workspace_bytes == 0
            else self._acquire(
                workspace_bytes,
                _LeaseProvenance(
                    AdmissionReplayPurpose.TASK_WORKSPACE,
                    task_id=task.task_id,
                ),
            )
        )

        new_alias_leases: dict[str, int] = {}
        for alias_id in admission.fresh_output_aliases:
            new_alias_leases[alias_id] = self._acquire_alias(
                task.task_id, alias_id, AdmissionReplayPurpose.TASK_OUTPUT
            )
        for alias_id in replacements:
            if self.alias_size[alias_id] == 0:
                continue
            if alias_id not in self.active_aliases:
                raise ValueError(
                    f"task {task.task_id} replaces nonresident alias {alias_id!r}"
                )
            new_alias_leases[alias_id] = self._acquire_alias(
                task.task_id,
                alias_id,
                AdmissionReplayPurpose.MUTATION_REPLACEMENT,
            )

        # The compiled task has now completed. Workspace is task-local, while
        # returned allocations become persistent object generations.
        if workspace_lease is not None:
            self._release(
                workspace_lease,
                _LeaseProvenance(
                    AdmissionReplayPurpose.TASK_WORKSPACE,
                    task_id=task.task_id,
                ),
            )
        self._publish_handoffs(task.task_id, handoffs)
        self._publish_replacements(task.task_id, replacements, new_alias_leases)
        for alias_id, lease_id in new_alias_leases.items():
            if alias_id not in replacement_aliases:
                if alias_id in self.active_aliases:
                    raise ValueError(
                        f"task {task.task_id} creates resident alias {alias_id!r}"
                    )
                self.active_aliases[alias_id] = lease_id
        self._apply_actions(task.task_id)

    def _validate_task_inputs(self, task: TaskSpec) -> None:
        required = dict.fromkeys(
            self.alias_by_object[object_id]
            for object_id in (
                *task.inputs,
                *(mutation.object_id for mutation in task.mutations),
            )
        )
        missing = sorted(
            alias_id
            for alias_id in required
            if self.alias_size[alias_id] != 0
            and alias_id not in self.active_aliases
        )
        if missing:
            raise ValueError(
                f"task {task.task_id} starts without execution aliases {missing}"
            )

    def _acquire_alias(
        self,
        task_id: str,
        alias_id: str,
        purpose: AdmissionReplayPurpose,
    ) -> int:
        return self._acquire(
            self.alias_size[alias_id],
            _LeaseProvenance(purpose, task_id=task_id, alias_group_id=alias_id),
        )

    def _publish_handoffs(
        self,
        task_id: str,
        handoffs: tuple[StorageHandoff, ...],
    ) -> None:
        for handoff in handoffs:
            source = handoff.source_alias_group_id
            destination = handoff.destination_alias_group_id
            if self.alias_size[destination] == 0:
                continue
            if source not in self.active_aliases:
                raise ValueError(
                    f"task {task_id} hands off nonresident alias {source!r}"
                )
            if destination in self.active_aliases:
                raise ValueError(
                    f"task {task_id} handoff destination "
                    f"{destination!r} is already resident"
                )
            lease_id = self.active_aliases.pop(source)
            self.active_aliases[destination] = lease_id
            self.handoff_releases.add((task_id, source))
            self.ownership_transitions.append(
                OwnershipTransition(
                    task_id,
                    OwnershipTransitionKind.STORAGE_HANDOFF,
                    destination,
                    lease_id,
                    source_alias_group_id=source,
                    source_lease_id=lease_id,
                )
            )

    def _publish_replacements(
        self,
        task_id: str,
        replacements: tuple[str, ...],
        new_alias_leases: Mapping[str, int],
    ) -> None:
        for alias_id in replacements:
            if self.alias_size[alias_id] == 0:
                continue
            old_lease = self.active_aliases[alias_id]
            new_lease = new_alias_leases[alias_id]
            self._release(
                old_lease,
                _LeaseProvenance(
                    AdmissionReplayPurpose.MUTATION_REPLACEMENT,
                    task_id=task_id,
                    alias_group_id=alias_id,
                ),
            )
            self.active_aliases[alias_id] = new_lease
            self.ownership_transitions.append(
                OwnershipTransition(
                    task_id,
                    OwnershipTransitionKind.MUTATION_REPLACEMENT,
                    alias_id,
                    new_lease,
                    source_alias_group_id=alias_id,
                    source_lease_id=old_lease,
                )
            )

    def _apply_actions(self, task_id: str) -> None:
        for action_index, action in self.actions_by_task.get(task_id, ()):
            alias_id = action.alias_group_id
            if action.kind is MemoryActionKind.RELEASE:
                if (task_id, alias_id) in self.handoff_releases:
                    continue
                lease_id = self._remove_active_alias(task_id, alias_id, action)
                self._release(
                    lease_id,
                    _LeaseProvenance(
                        AdmissionReplayPurpose.RELEASE,
                        task_id=task_id,
                        alias_group_id=alias_id,
                        action_index=action_index,
                    ),
                )
            elif action.kind is MemoryActionKind.OFFLOAD:
                lease_id = self._remove_active_alias(task_id, alias_id, action)
                dependency_id = self._new_dependency_id()
                self._append(
                    lease_id,
                    AdmissionReplayOperationKind.BEGIN_RETIREMENT,
                    _LeaseProvenance(
                        AdmissionReplayPurpose.EVICTION,
                        task_id=task_id,
                        alias_group_id=alias_id,
                        action_index=action_index,
                    ),
                    dependency_id=dependency_id,
                    dependency_expected=True,
                )
                self.pending_evictions.append(
                    _PendingEviction(
                        lease_id,
                        dependency_id,
                        task_id,
                        alias_id,
                        action_index,
                    )
                )
            else:
                if alias_id in self.active_aliases:
                    raise ValueError(
                        f"task {task_id} fetches resident alias {alias_id!r}"
                    )
                if self.alias_size[alias_id] == 0:
                    continue
                self.active_aliases[alias_id] = self._acquire(
                    self.alias_size[alias_id],
                    _LeaseProvenance(
                        AdmissionReplayPurpose.FETCH_DESTINATION,
                        task_id=task_id,
                        alias_group_id=alias_id,
                        action_index=action_index,
                    ),
                )

    def _remove_active_alias(
        self,
        task_id: str,
        alias_id: str,
        action: MemoryAction,
    ) -> int:
        try:
            return self.active_aliases.pop(alias_id)
        except KeyError as exc:
            raise ValueError(
                f"task {task_id} {action.kind.value}s nonresident alias "
                f"{alias_id!r}"
            ) from exc

    def _complete_pending_evictions(self) -> None:
        for pending in self.pending_evictions:
            self._append(
                pending.lease_id,
                AdmissionReplayOperationKind.COMPLETE_RETIREMENT,
                _LeaseProvenance(
                    AdmissionReplayPurpose.TERMINAL_COMPLETION,
                    task_id=pending.task_id,
                    alias_group_id=pending.alias_group_id,
                    action_index=pending.action_index,
                ),
                dependency_id=pending.dependency_id,
            )

    def _validate_final_execution_residency(self) -> None:
        required = {
            item.alias_group_id
            for item in self.schedule.final_residency
            if item.location is MemoryLocation.DEVICE
            and self.alias_size[item.alias_group_id] != 0
        }
        missing = sorted(required - self.active_aliases.keys())
        if missing:
            raise ValueError(
                f"admission replay lacks final execution aliases {missing}"
            )

    def _acquire(self, bytes_: int, provenance: _LeaseProvenance) -> int:
        lease_id = self._new_lease_id(provenance)
        self._append(
            lease_id,
            AdmissionReplayOperationKind.RESERVE,
            provenance,
            bytes_=bytes_,
            alignment=self.alignment,
        )
        self._append(
            lease_id,
            AdmissionReplayOperationKind.ACQUIRE_RESERVED,
            provenance,
        )
        return lease_id

    def _release(self, lease_id: int, provenance: _LeaseProvenance) -> None:
        self._append(lease_id, AdmissionReplayOperationKind.RELEASE, provenance)

    def _new_lease_id(self, provenance: _LeaseProvenance) -> int:
        lease_id = self.next_lease_id
        self.next_lease_id += 1
        self.lease_provenance[lease_id] = provenance
        return lease_id

    def _new_dependency_id(self) -> int:
        dependency_id = self.next_dependency_id
        self.next_dependency_id += 1
        return dependency_id

    def _append(
        self,
        lease_id: int,
        kind: AdmissionReplayOperationKind,
        provenance: _LeaseProvenance,
        *,
        bytes_: int = 0,
        alignment: int = 0,
        dependency_id: int | None = None,
        dependency_expected: bool = False,
    ) -> None:
        operation = PoolAdmissionOperation(
            sequence=len(self.operations),
            lease_id=lease_id,
            kind=kind,
            bytes=bytes_,
            alignment=alignment,
            dependency_id=dependency_id,
            dependency_expected=dependency_expected,
        )
        self.operations.append(
            AdmissionReplayStep(
                operation,
                provenance.purpose,
                task_id=provenance.task_id,
                alias_group_id=provenance.alias_group_id,
                action_index=provenance.action_index,
            )
        )


def replay_admission(
    program: Program,
    schedule: MemorySchedule,
    *,
    execution_pool_bytes: int,
    selections: tuple[RecomputationSelection, ...] = (),
    topology: AdmissionTopology | None = None,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
    alignment: int = 256,
) -> AdmissionReplay:
    """Certify one schedule's task-boundary geometry with no timing inputs."""

    if execution_pool_bytes < 0:
        raise ValueError("execution pool bytes must be non-negative")
    if alignment <= 0:
        raise ValueError("admission replay alignment must be positive")
    schedule.validate(program, selections)
    selected_topology = topology or build_admission_topology(
        program,
        execution_pool_bytes=execution_pool_bytes,
        object_capacity_bytes=execution_pool_bytes,
        output_bindings=output_bindings,
        alignment=alignment,
    )
    selected_topology.validate(program)
    if selected_topology.pool_capacity_bytes != execution_pool_bytes:
        raise ValueError(
            "admission topology and replay capacities differ: "
            f"topology={selected_topology.pool_capacity_bytes}, "
            f"replay={execution_pool_bytes}"
        )
    builder = _AdmissionScriptBuilder(
        program,
        schedule,
        selections,
        selected_topology,
    )
    (
        annotated_operations,
        ownership_transitions,
        pending_evictions,
        final_aliases,
        workspace_bytes_by_task,
        lease_count,
        dependency_count,
    ) = builder.build()
    operations = tuple(item.operation for item in annotated_operations)
    pool = run_admission_replay(
        selected_topology.pool_capacity_bytes,
        operations,
        lease_count=lease_count,
        dependency_count=dependency_count,
        minimum_alignment=selected_topology.minimum_alignment,
    )
    pending_by_lease = {item.lease_id: item for item in pending_evictions}
    provenance_by_lease = builder.lease_provenance
    dependencies = tuple(
        _resolve_dependency(
            item,
            pending_by_lease=pending_by_lease,
            provenance_by_lease=provenance_by_lease,
        )
        for item in pool.dependencies
    )
    digest = _compatibility_digest(
        program,
        schedule,
        selections,
        annotated_operations,
        ownership_transitions,
        pool.decision_digest,
    )
    return AdmissionReplay(
        pool,
        annotated_operations,
        ownership_transitions,
        dependencies,
        final_aliases,
        workspace_bytes_by_task,
        digest,
    )


def _resolve_dependency(
    dependency: AdmissionReuseDependency,
    *,
    pending_by_lease: Mapping[int, _PendingEviction],
    provenance_by_lease: Mapping[int, _LeaseProvenance],
) -> CausalAdmissionDependency:
    predecessor_lease_id = dependency.predecessor_lease_id
    successor_lease_id = dependency.successor_lease_id
    predecessor = pending_by_lease[predecessor_lease_id]
    successor = provenance_by_lease[successor_lease_id]
    return CausalAdmissionDependency(
        dependency_id=dependency.dependency_id,
        predecessor_lease_id=predecessor_lease_id,
        predecessor_task_id=predecessor.task_id,
        predecessor_alias_group_id=predecessor.alias_group_id,
        predecessor_action_index=predecessor.action_index,
        successor_lease_id=successor_lease_id,
        successor_task_id=successor.task_id,
        successor_alias_group_id=successor.alias_group_id,
        successor_action_index=successor.action_index,
        consumer_operation_index=dependency.consumer_operation_index,
    )


def _compatibility_digest(
    program: Program,
    schedule: MemorySchedule,
    selections: tuple[RecomputationSelection, ...],
    operations: tuple[AdmissionReplayStep, ...],
    transitions: tuple[OwnershipTransition, ...],
    decision_digest: int,
) -> str:
    payload = {
        "program": program.digest,
        "schedule": schedule.digest,
        "selections": [item.to_dict() for item in selections],
        "operations": [
            {
                "sequence": item.operation.sequence,
                "lease_id": item.operation.lease_id,
                "kind": int(item.operation.kind),
                "bytes": item.operation.bytes,
                "alignment": item.operation.alignment,
                "dependency_id": item.operation.dependency_id,
                "dependency_expected": item.operation.dependency_expected,
                "purpose": item.purpose.value,
                "task_id": item.task_id,
                "alias_group_id": item.alias_group_id,
                "action_index": item.action_index,
            }
            for item in operations
        ],
        "ownership_transitions": [
            {
                "task_id": item.task_id,
                "kind": item.kind.value,
                "destination_alias_group_id": item.destination_alias_group_id,
                "destination_lease_id": item.destination_lease_id,
                "source_alias_group_id": item.source_alias_group_id,
                "source_lease_id": item.source_lease_id,
            }
            for item in transitions
        ],
        "decision_digest": decision_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = [
    "AdmissionReplay",
    "AdmissionReplayPurpose",
    "AdmissionReplayStep",
    "CausalAdmissionDependency",
    "OwnershipTransition",
    "OwnershipTransitionKind",
    "replay_admission",
]
