"""Exact production-``MemoryPool`` admission interpreter."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum

from shadowspill._status import Status

from ._admission_capi import (
    ABI_VERSION,
    NO_ID,
    CAdmissionReplayDecision,
    CAdmissionReplayLiveLease,
    CAdmissionReplayOperation,
    CAdmissionReplayProgram,
    CAdmissionReplayResult,
    CAdmissionReuseDependency,
    load_admission_replay_library,
)
from .admission import AdmissionError


class AdmissionReplayOperationKind(IntEnum):
    ACQUIRE = 0
    BEGIN_RETIREMENT = 1
    PUBLISH_DEPENDENCY = 2
    RESERVE = 3
    ACQUIRE_RESERVED = 4
    COMPLETE_RETIREMENT = 5
    RELEASE = 6


class AdmissionReplayLeaseState(IntEnum):
    FREE = 0
    IN_USE = 1
    RETIRE_PENDING = 2
    RESERVED = 3
    SUCCESSOR_RESERVED = 4
    PREDECESSOR_TRANSFERRED = 5


@dataclass(frozen=True, slots=True)
class AdmissionReplayOperation:
    sequence: int
    lease_id: int
    kind: AdmissionReplayOperationKind
    bytes: int = 0
    alignment: int = 0
    dependency_id: int | None = None
    dependency_expected: bool = False

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.lease_id < 0:
            raise ValueError(
                "admission sequence and lease ID must be non-negative"
            )
        if self.bytes < 0 or self.alignment < 0:
            raise ValueError("admission geometry must be non-negative")
        if self.dependency_id is not None and self.dependency_id < 0:
            raise ValueError("admission dependency ID must be non-negative")
        if not isinstance(self.kind, AdmissionReplayOperationKind):
            raise TypeError(
                "admission operation kind must be AdmissionReplayOperationKind"
            )


@dataclass(frozen=True, slots=True)
class AdmissionReplayDecision:
    operation_index: int
    sequence: int
    lease_id: int
    predecessor_lease_id: int | None
    dependency_id: int | None
    offset: int
    requested_bytes: int
    charged_bytes: int
    physical_bytes_delta: int
    resulting_state: AdmissionReplayLeaseState


@dataclass(frozen=True, slots=True)
class AdmissionReuseDependency:
    predecessor_lease_id: int
    successor_lease_id: int
    dependency_id: int
    consumer_operation_index: int


@dataclass(frozen=True, slots=True)
class AdmissionReplayResult:
    capacity_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    peak_fragmentation_bytes: int
    final_allocated_bytes: int
    final_reserved_bytes: int
    final_largest_free_range_bytes: int
    decision_digest: int
    decisions: tuple[AdmissionReplayDecision, ...]
    dependencies: tuple[AdmissionReuseDependency, ...]


def run_admission_replay(
    capacity_bytes: int,
    operations: tuple[AdmissionReplayOperation, ...],
    *,
    lease_count: int,
    dependency_count: int,
    minimum_alignment: int = 256,
    large_request_threshold_bytes: int = 0,
) -> AdmissionReplayResult:
    """Replay an ordered script through the exact production MemoryPool core."""

    if (
        capacity_bytes < 0
        or lease_count < 0
        or dependency_count < 0
        or large_request_threshold_bytes < 0
    ):
        raise ValueError("admission capacities and counts must be non-negative")
    if minimum_alignment <= 0:
        raise ValueError("minimum alignment must be positive")
    operation_buffer = (CAdmissionReplayOperation * max(1, len(operations)))(
        *(
            CAdmissionReplayOperation(
                operation.sequence,
                operation.lease_id,
                NO_ID if operation.dependency_id is None else operation.dependency_id,
                operation.bytes,
                operation.alignment,
                int(operation.kind),
                int(operation.dependency_expected),
            )
            for operation in operations
        )
    )
    decision_buffer = (CAdmissionReplayDecision * max(1, len(operations)))()
    dependency_buffer = (CAdmissionReuseDependency * max(1, len(operations)))()
    live_lease_buffer = (CAdmissionReplayLiveLease * max(1, lease_count))()
    program = CAdmissionReplayProgram(
        ABI_VERSION,
        capacity_bytes,
        minimum_alignment,
        large_request_threshold_bytes,
        lease_count,
        dependency_count,
        operation_buffer,
        len(operations),
    )
    result = CAdmissionReplayResult(
        decisions=decision_buffer,
        decision_capacity=len(operations),
        dependencies=dependency_buffer,
        dependency_capacity=len(operations),
        live_leases=live_lease_buffer,
        live_lease_capacity=lease_count,
    )
    library = load_admission_replay_library()
    status = int(
        library.shadowspill_admission_replay_run(
            ctypes.byref(program), ctypes.byref(result)
        )
    )
    if status == Status.REPLAY_INFEASIBLE:
        raise AdmissionError(
            "production MemoryPool replay cannot satisfy operation "
            f"{int(result.error_operation_index)} for lease "
            f"{int(result.error_lease_id)}: requested "
            f"{int(result.error_requested_bytes)} bytes, free "
            f"{int(result.error_free_bytes)}, largest range "
            f"{int(result.error_largest_free_range_bytes)}; "
            f"{int(result.live_lease_count)} physical leases remain live",
            kind="memory_pool_fragmentation",
            required_bytes=int(result.error_requested_bytes),
            capacity_bytes=capacity_bytes,
            position=int(result.error_operation_index),
            free_bytes=int(result.error_free_bytes),
            largest_free_range_bytes=int(result.error_largest_free_range_bytes),
            live_lease_evidence=tuple(
                (
                    int(item.lease_id),
                    int(item.offset),
                    int(item.requested_bytes),
                    int(item.charged_bytes),
                    int(item.state),
                )
                for item in live_lease_buffer[: int(result.live_lease_count)]
            ),
        )
    if status != 0:
        description = library.shadowspill_admission_replay_status_string(
            status
        ).decode()
        raise ValueError(
            f"MemoryPool replay failed at operation "
            f"{int(result.error_operation_index)} with {description}"
        )
    decisions = tuple(
        AdmissionReplayDecision(
            int(item.operation_index),
            int(item.sequence),
            int(item.lease_id),
            _optional_id(int(item.predecessor_lease_id)),
            _optional_id(int(item.dependency_id)),
            int(item.offset),
            int(item.requested_bytes),
            int(item.charged_bytes),
            int(item.physical_bytes_delta),
            AdmissionReplayLeaseState(int(item.resulting_state)),
        )
        for item in decision_buffer[: int(result.decision_count)]
    )
    dependencies = tuple(
        AdmissionReuseDependency(
            int(item.predecessor_lease_id),
            int(item.successor_lease_id),
            int(item.dependency_id),
            int(item.consumer_operation_index),
        )
        for item in dependency_buffer[: int(result.dependency_result_count)]
    )
    return AdmissionReplayResult(
        capacity_bytes=capacity_bytes,
        peak_allocated_bytes=int(result.peak_allocated_bytes),
        peak_reserved_bytes=int(result.peak_reserved_bytes),
        peak_fragmentation_bytes=int(result.peak_fragmentation_bytes),
        final_allocated_bytes=int(result.final_allocated_bytes),
        final_reserved_bytes=int(result.final_reserved_bytes),
        final_largest_free_range_bytes=int(result.final_largest_free_range_bytes),
        decision_digest=int(result.decision_digest),
        decisions=decisions,
        dependencies=dependencies,
    )


def _optional_id(value: int) -> int | None:
    return None if value == NO_ID else value


__all__ = [
    "AdmissionReplayDecision",
    "AdmissionReplayLeaseState",
    "AdmissionReplayOperation",
    "AdmissionReplayOperationKind",
    "AdmissionReplayResult",
    "AdmissionReuseDependency",
    "run_admission_replay",
]
