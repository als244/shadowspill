"""CPU accounting for fixed reservations shared by one execution worker."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from shadowspill.artifacts._values import digest, nonnegative_int, plain
from shadowspill.resources.inventory import resource_inventory

if TYPE_CHECKING:
    from shadowspill.artifacts.plan import Plan
    from shadowspill.artifacts.pool_memory import PoolMemoryProfile
    from shadowspill.communication import CommunicationInfo
    from shadowspill.communication.requests import CollectiveRequest


def _communicator_coverage(
    plan: "Plan",
) -> tuple[str, frozenset["CollectiveRequest"]] | None:
    """One measured startup's scope and exact warmed collective forms.

    A shared worker creates each process group once and warms the union of its
    Plans' requests. A calibration covering that union already observed those
    resources. Incomplete or incompatible records retain additive accounting.
    """
    from shadowspill.communication.requests import (
        CollectiveRequest,
        collective_requests,
    )

    info = plan.communication_info
    if info.metadata.get("memory_kind") != "measured" or not info.measurements:
        return None
    scope = info.metadata["memory_scope"]
    if plain(scope.get("backend_setup", {})) != plain(
        info.metadata.get("backend_setup", {})
    ) or plain(scope["topology"]) != plain(info.topology):
        return None
    requests = set()
    for row in info.measurements:
        if row.get("memory", {}).get("memory_kind") != "measured" or any(
            plain(row.get(key, {})) != plain(scope.get(key, {}))
            for key in ("setup", "topology", "backend_setup")
        ):
            return None
        requests.add(
            CollectiveRequest(
                row["operation"],
                tuple(row["ranks"]),
                row["tensor_bytes"],
                row["dtype"],
                row.get("inplace"),
            )
        )
    if {request.ranks for request in requests} != {
        tuple(group) for group in scope["groups"]
    } or not set(collective_requests(plan.program, None)) <= requests:
        return None
    return digest(scope), frozenset(requests)


def external_baseline(plans: Sequence["Plan"]) -> int:
    """Maximum measured process-external footprint, excluding in-arena state.

    A legacy profile without this component retains observed-baseline startup.
    Missing decomposition is not a new memory measurement.
    """
    return max(
        (p.profiled_program.metadata.get("external_device_bytes") or 0 for p in plans),
        default=0,
    )


def communicator_reservation(plans: Sequence["Plan"]) -> int:
    """Reserve measured startup for the worker's shared process groups.

    This is the startup reservation, not collective task/transfer workspace.
    Keep CPU admission and runtime headroom on the same identity/deduplication
    rule. Compatible measured calibrations share their largest reservation only
    when one covers every warmed request in their union. Otherwise add them.
    """
    if not plans:
        return 0
    parallelism = plans[0].program.parallelism
    if parallelism is None or parallelism.world_size <= 1:
        return 0
    calibrations: dict[str, list[Plan]] = {}
    for plan in plans:
        calibrations.setdefault(digest(plan.communication_info.to_dict()), []).append(
            plan
        )
    measured: dict[
        str, list[tuple[frozenset[CollectiveRequest], CommunicationInfo]]
    ] = {}
    total = 0
    for users in calibrations.values():
        info = users[0].communication_info
        coverage = [_communicator_coverage(plan) for plan in users]
        if any(item is None for item in coverage):
            total += info.fixed_device_bytes or 0
            continue
        first = coverage[0]
        assert first is not None
        scope, requests = first
        measured.setdefault(scope, []).append((requests, info))
    for records in measured.values():
        union = frozenset().union(*(requests for requests, _ in records))
        sizes = [info.fixed_device_bytes or 0 for _, info in records]
        total += (
            max(sizes)
            if any(requests == union for requests, _ in records)
            else sum(sizes)
        )
    return total


def shared_fixed_overhead(plans: Sequence["Plan"]) -> int:
    """Account for maxima that need not occur in the same individual Plan.

    External baseline and headroom each take their worker-wide maximum. The
    retained resources follow their actual sharing identities; unclassified
    retained state is reserved for every separately bound path. Communication
    scratch is sequential and takes its maximum. Communicator startup follows the
    same shared-calibration reservation as execution; it is not counted again
    in that remainder. Invocation paths, including continuing gradients, belong
    in ``plans`` even when there are no additional named Plans.

    Legacy undecomposed fixed costs remain additive. This can conservatively
    over-reserve; it never invents a measured split or discounts unknown costs.
    """
    headroom = 0
    scratch = 0
    communicator = communicator_reservation(plans)
    for plan in plans:
        profile = plan.profiled_program
        margin = profile.metadata.get("external_headroom_bytes") or 0
        communication = plan._communication_memory[0]
        fixed = (
            (plan.communication_info.fixed_device_bytes or 0) if communication else 0
        )
        if fixed > communication:
            raise ValueError("communicator reservation exceeds communication overhead")
        communicator = max(communicator, fixed)
        headroom = max(headroom, margin)
        scratch = max(scratch, communication - fixed)
    return (
        external_baseline(plans)
        + headroom
        + communicator
        + scratch
        + retained_reservation(plans)
        + pool_process_reservation(plans)
    )


def retained_reservation(plans: Sequence["Plan"]) -> int:
    """Bound the union loaded at startup, not the largest individual profile.

    Identical complete resource/stream/preparation inventories can share their
    maximum measured bytes. Partially overlapping inventories are additive, a
    conservative bound without invented per-resource measurements. Remaining
    retained allocations belong to each independently restored execution path.
    """
    resource_inventory(plan.program for plan in plans)
    shared: dict[tuple[str, str], int] = {}
    private = 0
    for plan in plans:
        program, profile = plan.program, plan.profiled_program
        metadata = profile.metadata
        external = metadata.get("external_device_bytes") or 0
        margin = metadata.get("external_headroom_bytes") or 0
        remaining = profile.fixed_device_bytes - external - margin
        if remaining < 0:
            raise ValueError("external baseline and headroom exceed fixed overhead")
        for kind, field, identity in (
            (
                "streams",
                "stream_workspace_bytes",
                metadata.get("resource_streams"),
            ),
            (
                "resources",
                "provider_resource_bytes",
                {
                    name: {
                        "description": description.to_dict(),
                        "source": program.bindings["resource_sources"][name],
                    }
                    for name, description in program.resources.items()
                },
            ),
            (
                "preparations",
                "prepared_operator_retained_bytes",
                program.bindings.get("operator_preparation"),
            ),
        ):
            size = metadata.get(field, 0)
            nonnegative_int(size, field)
            if not size:
                continue
            if not identity:
                raise ValueError(f"{field} lacks its retained inventory")
            remaining -= size
            key = kind, digest(identity)
            shared[key] = max(shared.get(key, 0), size)
        if remaining < 0:
            raise ValueError("retained components exceed fixed overhead")
        private += remaining
    return private + sum(shared.values())


def shared_pool_memory(plans: Sequence["Plan"]) -> "PoolMemoryProfile | None":
    """One identical observation record for all invocation paths of a worker.

    Refuse mixing uncalibrated and calibrated Plans. Replanning with one shared
    calibration is cheap and avoids inventing a composite acquisition history.
    """
    profiles = [p.pool_memory for p in plans]
    if not profiles:
        return None
    if any(p != profiles[0] for p in profiles):
        raise ValueError("shared Plans require identical pool memory evidence")
    return profiles[0]


def pool_process_reservation(plans: Sequence["Plan"]) -> int:
    profile = shared_pool_memory(plans)
    return 0 if profile is None else profile.process_reservation_bytes
