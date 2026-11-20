"""Exact production-``MemoryPool`` admission interpreter."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum

from shadowspill.status import ABI_VERSION, Status

from .admission_capi import (
    NO_ID,
    CAdmissionReplayDecision,
    CAdmissionReplayLiveLease,
    CAdmissionReplayOperation,
    CAdmissionReplayProgram,
    CAdmissionReplayResult,
    CAdmissionReuseDependency,
    admission_replay_api,
    optional_id,
)


class PhysicalAdmissionError(ValueError):
    """Raised when a replayed operation finds no range of the pool that fits.

    Parameters
    ----------
    message : str
        The error message.
    kind : str
        What failed; `run_admission_replay` gives
        ``"memory_pool_fragmentation"``.
    required_bytes : int
        Bytes the failing operation asked for.
    capacity_bytes : int
        The pool's capacity, in bytes.
    position : int, optional
        Position of the failing operation in the script.
    free_bytes : int, optional
        Free bytes in the pool at that operation.
    largest_free_range_bytes : int, optional
        The pool's largest free range at that operation, in bytes.
    live_lease_evidence : tuple of tuple of int, default ()
        Every lease holding a range at that operation, in offset order, as
        ``(lease_id, offset, requested_bytes, charged_bytes, state)``, where
        ``state`` is an `AdmissionReplayLeaseState` value.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        required_bytes: int,
        capacity_bytes: int,
        position: int | None = None,
        free_bytes: int | None = None,
        largest_free_range_bytes: int | None = None,
        live_lease_evidence: tuple[tuple[int, int, int, int, int], ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.position = position
        self.free_bytes = free_bytes
        self.largest_free_range_bytes = largest_free_range_bytes
        self.live_lease_evidence = live_lease_evidence


class AdmissionReplayOperationKind(IntEnum):
    """What one replayed operation does to its lease.

    Mirrors ``ShadowSpillAdmissionReplayOperationKind``. The kinds are
    transitions of ownership in the pool, not transfers.

    Attributes
    ----------
    ACQUIRE : int
        Take a range of ``bytes`` for the lease. When no free range fits, the
        fewest retiring leases, in the order their retirements began, whose
        ranges make room are released early, and the lease records a reuse
        dependency on each.
    BEGIN_RETIREMENT : int
        Start retiring an in-use lease on the dependency ``dependency_id``.
        It holds its range until the retirement completes.
    PUBLISH_DEPENDENCY : int
        Give a retiring lease the dependency it was promised.
    RESERVE : int
        Claim a range for the lease ahead of its use: a free range, else the
        whole range of a retiring lease, which this lease then succeeds, else
        room made as ``ACQUIRE`` makes it.
    ACQUIRE_RESERVED : int
        Take up the range the lease reserved. A successor takes over its
        predecessor's range and records the dependency it must wait on.
    COMPLETE_RETIREMENT : int
        Finish a retirement on the dependency ``dependency_id`` and give its
        range back.
    RELEASE : int
        Give the lease's range back at once, with nothing to wait on.
    """

    ACQUIRE = 0
    BEGIN_RETIREMENT = 1
    PUBLISH_DEPENDENCY = 2
    RESERVE = 3
    ACQUIRE_RESERVED = 4
    COMPLETE_RETIREMENT = 5
    RELEASE = 6


class AdmissionReplayLeaseState(IntEnum):
    """The state a replayed operation leaves its lease in.

    Mirrors ``ShadowSpillAdmissionReplayLeaseState``; each
    `AdmissionReplayDecision` reports it as ``resulting_state``.

    Attributes
    ----------
    FREE : int
        Holds no range.
    IN_USE : int
        Holds its range and is in use.
    RETIRE_PENDING : int
        Retiring: holds its range until its retirement completes.
    RESERVED : int
        Holds a range claimed ahead of use, which ``ACQUIRE_RESERVED`` takes
        up.
    SUCCESSOR_RESERVED : int
        Promised the whole range of a retiring lease. Taking it up takes the
        range over; if the retiring lease finishes first, the range passes to
        this one as ``RESERVED``.
    PREDECESSOR_TRANSFERRED : int
        A retiring lease whose range its successor has taken over. It holds
        no range, and completing its retirement frees it.
    """

    FREE = 0
    IN_USE = 1
    RETIRE_PENDING = 2
    RESERVED = 3
    SUCCESSOR_RESERVED = 4
    PREDECESSOR_TRANSFERRED = 5


@dataclass(frozen=True, slots=True)
class AdmissionReplayOperation:
    """One operation of an admission replay script.

    Parameters
    ----------
    sequence : int
        Ordering key, at least zero. Operations replay in the order given,
        and their sequences must not decrease.
    lease_id : int
        The lease the operation acts on, from zero to ``lease_count`` less
        one.
    kind : AdmissionReplayOperationKind
        What the operation does.
    bytes : int, default 0
        Size requested, in bytes; positive for ``ACQUIRE`` and ``RESERVE``.
    alignment : int, default 0
        Alignment requested, in bytes; positive for ``ACQUIRE`` and
        ``RESERVE``. A smaller one than the replay's ``minimum_alignment`` is
        raised to it.
    dependency_id : int, optional
        The dependency a retirement waits on, from zero to
        ``dependency_count`` less one. ``BEGIN_RETIREMENT``,
        ``PUBLISH_DEPENDENCY`` and ``COMPLETE_RETIREMENT`` need one.
    dependency_expected : bool, default False
        For ``BEGIN_RETIREMENT``: the dependency is promised rather than
        present, and a later ``PUBLISH_DEPENDENCY`` gives it.

    Raises
    ------
    ValueError
        If the sequence, lease ID, size, alignment or dependency ID is
        negative.
    TypeError
        If ``kind`` is not an `AdmissionReplayOperationKind`.
    """

    sequence: int
    lease_id: int
    kind: AdmissionReplayOperationKind
    bytes: int = 0
    alignment: int = 0
    dependency_id: int | None = None
    dependency_expected: bool = False

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.lease_id < 0:
            raise ValueError("admission sequence and lease ID must be non-negative")
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
    """What one replayed operation did.

    `AdmissionReplayResult.decisions` holds one per operation, in order.

    Attributes
    ----------
    operation_index : int
        Position of the operation in the script.
    sequence : int
        The operation's ``sequence``.
    lease_id : int
        The operation's lease.
    predecessor_lease_id : int or None
        For ``RESERVE`` and ``ACQUIRE_RESERVED``, the retiring lease whose
        range this lease succeeds; None otherwise.
    dependency_id : int or None
        For ``ACQUIRE_RESERVED`` by a successor, the dependency it must wait
        on before using the range; None otherwise.
    offset : int
        Byte offset of the lease's range after the operation, or, for an
        operation that gave the range up, of the range it held.
    requested_bytes : int
        Bytes the lease asked for, read the same way.
    charged_bytes : int
        Bytes its range occupies, read the same way.
    physical_bytes_delta : int
        Change in the pool's allocated bytes the operation caused; negative
        when it freed bytes.
    resulting_state : AdmissionReplayLeaseState
        The lease's state after the operation.
    """

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
    """A wait a replayed lease owes before it uses a reused range.

    The lease ``successor_lease_id`` takes over bytes that
    ``predecessor_lease_id`` held while still retiring, so its consumer must
    wait until the dependency ``dependency_id`` completes.
    `AdmissionReplayResult.dependencies` lists them in the order they arose.

    Attributes
    ----------
    predecessor_lease_id : int
        The retiring lease whose range is reused.
    successor_lease_id : int
        The lease that reuses it.
    dependency_id : int
        The predecessor's retirement dependency.
    consumer_operation_index : int
        Position of the operation that made the reuse.
    """

    predecessor_lease_id: int
    successor_lease_id: int
    dependency_id: int
    consumer_operation_index: int


@dataclass(frozen=True, slots=True)
class AdmissionReplayResult:
    """The outcome of `run_admission_replay`.

    Attributes
    ----------
    capacity_bytes : int
        The replayed pool's capacity, in bytes.
    peak_allocated_bytes : int
        Most bytes the pool's ranges held at once.
    peak_reserved_bytes : int
        Most bytes held at once by reserved leases not yet taken up.
    peak_fragmentation_bytes : int
        Most free bytes that lay outside the largest free range after any
        operation: free bytes no single request could use.
    final_allocated_bytes : int
        Bytes the pool's ranges held after the last operation.
    final_reserved_bytes : int
        Bytes reserved and not yet taken up after the last operation.
    final_largest_free_range_bytes : int
        The largest free range after the last operation, in bytes.
    decision_digest : int
        A 64-bit digest of every decision, equal for replays that make the
        same decisions.
    decisions : tuple of AdmissionReplayDecision
        One per operation, in order.
    dependencies : tuple of AdmissionReuseDependency
        Every reuse dependency, in the order the replay recorded them.
    """

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
    """Replay a script of pool operations through the runtime's own pool policy.

    The pool holds offsets only: no device and no memory are used. Lease and
    dependency IDs are indices from zero, bounded by ``lease_count`` and
    ``dependency_count``.

    Parameters
    ----------
    capacity_bytes : int
        Capacity of the replayed pool, in bytes. At least zero.
    operations : sequence of AdmissionReplayOperation
        The script, with ``sequence`` never decreasing.
    lease_count : int
        Number of leases the script names. At least zero.
    dependency_count : int
        Number of dependencies the script names. At least zero.
    minimum_alignment : int, default 256
        Smallest alignment of any range, in bytes. Positive.
    large_request_threshold_bytes : int, default 0
        Size, in bytes, from which a request takes the high end of the free
        range that fits it best rather than the low end. Zero places every
        request low.

    Returns
    -------
    AdmissionReplayResult
        Every decision, every reuse dependency, and the pool's peaks and
        final state.

    Raises
    ------
    PhysicalAdmissionError
        If an operation finds no range that fits. Its ``position``,
        ``required_bytes``, ``free_bytes``, ``largest_free_range_bytes`` and
        ``live_lease_evidence`` describe the pool at that operation. It is a
        `ValueError`.
    ValueError
        If a capacity or count is negative, ``minimum_alignment`` is not
        positive, or the script is invalid: an ID out of range, a decreasing
        sequence, a zero size or alignment where one is needed, or an operation
        its lease's state does not allow.
    """

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
    library = admission_replay_api()
    status = int(
        library.shadowspill_admission_replay_run(
            ctypes.byref(program), ctypes.byref(result)
        )
    )
    if status == Status.REPLAY_INFEASIBLE:
        raise PhysicalAdmissionError(
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
        description = library.shadowspill_status_string(status).decode()
        raise ValueError(
            f"MemoryPool replay failed at operation "
            f"{int(result.error_operation_index)} with {description}"
        )
    decisions = tuple(
        AdmissionReplayDecision(
            int(item.operation_index),
            int(item.sequence),
            int(item.lease_id),
            optional_id(int(item.predecessor_lease_id)),
            optional_id(int(item.dependency_id)),
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


__all__ = [
    "AdmissionReplayDecision",
    "AdmissionReplayLeaseState",
    "AdmissionReplayOperation",
    "AdmissionReplayOperationKind",
    "AdmissionReplayResult",
    "AdmissionReuseDependency",
    "PhysicalAdmissionError",
    "run_admission_replay",
]
