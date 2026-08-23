"""Lease lifetimes for one schedule, from the library.

The library joins the operations a schedule implies against the timings the
simulator measured, and reports one record per lease: the four numbers
placement reads, and beside them the identity a certificate needs. Both come
back as arrays, and the caller reads whichever it needs — a measurement wants
only the placeable prefix, so it never touches the identities at all.

Fixed leases occupy the prefix `[0, fixed_count)`; the caller-owned dynamic
ones follow, so placement runs on the prefix without a copy.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from shadowspill._status import ABI_VERSION
from shadowspill.simulator import SimulationResult
from shadowspill.simulator._indexed import IntervalArrays

from ._admission import IndexedAdmissionTopology
from ._capi import (
    CLeaseIdentity,
    CLeaseLifetime,
    CLeaseLifetimeProblem,
    CLeaseLifetimeResult,
    load_planner_library,
)
from ._operations import AdmissionOperations


@dataclass(frozen=True, slots=True)
class LeaseLifetimes:
    """One schedule's leases, partitioned fixed-first.

    `lifetimes` and `identities` are the library's own arrays, indexed alike.
    Reading either is a slice, not a decode, so a caller pays only for what it
    looks at.
    """

    lifetimes: ctypes.Array[CLeaseLifetime]
    identities: ctypes.Array[CLeaseIdentity]
    allocation_step_leases: ctypes.Array[ctypes.c_uint64]
    alias_leases: ctypes.Array[ctypes.c_uint64]
    count: int
    fixed_count: int


def build_lease_lifetimes(
    operations: AdmissionOperations,
    admission: IndexedAdmissionTopology,
    simulation: SimulationResult,
    *,
    dynamic_aliases: tuple[int, ...] = (),
) -> LeaseLifetimes:
    """Resolve every lease to a lifetime, and split off the dynamic ones.

    `dynamic_aliases` names caller-owned terminal aliases by index; the lease
    each one ends the step holding is moved out of the fixed prefix. Raising
    here means one of them never reached a final lease, which is a caller
    error rather than a planning outcome.
    """

    intervals = simulation.interval_arrays
    if not isinstance(intervals, IntervalArrays):
        raise ValueError(
            "lease lifetimes need the simulator's own intervals; "
            "this result did not come from it"
        )

    topology = admission.value
    steps = topology.task_allocation_offsets[topology.task_count]
    count = operations.lease_count
    lifetimes = (CLeaseLifetime * max(count, 1))()
    identities = (CLeaseIdentity * max(count, 1))()
    step_leases = (ctypes.c_uint64 * max(steps, 1))()
    alias_leases = (ctypes.c_uint64 * max(topology.alias_count, 1))()
    result = CLeaseLifetimeResult(
        lifetimes=lifetimes,
        identities=identities,
        allocation_step_leases=step_leases,
        alias_leases=alias_leases,
        lifetime_count=0,
        fixed_count=0,
    )
    problem = CLeaseLifetimeProblem(
        abi_version=ABI_VERSION,
        operations=ctypes.pointer(operations.arrays.operations),
        admission=ctypes.pointer(admission.value),
        schedule=ctypes.pointer(operations.arrays.schedule),
        task_intervals=intervals.task_intervals,
        task_interval_count=intervals.task_interval_count,
        transfer_intervals=intervals.transfer_intervals,
        transfer_interval_count=intervals.transfer_interval_count,
        makespan_ns=simulation.makespan_ns,
        dynamic_aliases=(ctypes.c_uint32 * max(len(dynamic_aliases), 1))(
            *dynamic_aliases
        ),
        dynamic_alias_count=len(dynamic_aliases),
    )
    status = int(
        load_planner_library().shadowspill_build_lease_lifetimes(
            ctypes.byref(problem), ctypes.byref(result)
        )
    )
    if status != 0:
        raise RuntimeError(
            f"building lease lifetimes failed with planner status {status}"
        )
    return LeaseLifetimes(
        lifetimes=lifetimes,
        identities=identities,
        allocation_step_leases=step_leases,
        alias_leases=alias_leases,
        count=int(result.lifetime_count),
        fixed_count=int(result.fixed_count),
    )


__all__ = ["LeaseLifetimes", "build_lease_lifetimes"]
