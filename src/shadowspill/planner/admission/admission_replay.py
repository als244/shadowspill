"""Cross-task physical admission through the production ``MemoryPool``.

This module combines exact task-allocation evidence with persistent object
generations and ordered memory actions. It translates that causal script into
the same production ``MemoryPool`` decisions used by admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shadowspill.runtime import (
    AdmissionReplayOperation as PoolAdmissionOperation,
)
from shadowspill.runtime import (
    AdmissionReplayResult,
)


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
    predecessor_purpose: AdmissionReplayPurpose
    predecessor_alias_group_id: str | None
    predecessor_action_index: int | None
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


__all__ = [
    "AdmissionReplay",
    "AdmissionReplayPurpose",
    "AdmissionReplayStep",
    "CausalAdmissionDependency",
    "OwnershipTransition",
    "OwnershipTransitionKind",
]
