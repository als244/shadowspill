"""Price compound transfers without changing their runtime execution boundaries."""

from dataclasses import replace

from shadowspill.artifacts.program import Program
from shadowspill.communication import CommunicationInfo
from shadowspill.communication.collectives import matching_measurements
from shadowspill.communication.requests import CollectiveRequest
from shadowspill.communication.task_costs import task_collective_cost
from shadowspill.communication.transfers import transfer_collectives
from shadowspill.ir import ShadowSpillProgram, TransferPhase, TransferSpec


def host_phases(
    info: CommunicationInfo, direction: str, size: int
) -> tuple[TransferPhase, ...]:
    latency = round(getattr(info, f"{direction}_latency_seconds") * 1e9)
    return (
        ()
        if not latency
        else (
            TransferPhase(
                "timed",
                duration_ns=latency,
                resource=f"host_{direction}",
                timing_class="host_latency",
            ),
        )
    ) + (TransferPhase("copy", bytes=size),)


def communication_transfers(
    source: Program, costed: ShadowSpillProgram, info: CommunicationInfo
) -> ShadowSpillProgram:
    weights = source.bindings.get("weight_shards", {})
    gradients = source.bindings.get("gradient_shards", {})
    requests = transfer_collectives(source.bindings)
    aliases = []
    for alias in costed.alias_groups:
        key = alias.alias_group_id
        record = weights.get(key, gradients.get(key))
        if record is None:
            if info.fetch_latency_seconds or info.evict_latency_seconds:
                alias = replace(
                    alias,
                    transfer=TransferSpec(
                        alias.size_bytes,
                        host_phases(info, "fetch", alias.size_bytes),
                        host_phases(info, "evict", alias.size_bytes),
                    ),
                )
        else:
            if alias.size_bytes != record["padded_bytes"]:
                raise ValueError("shard geometry disagrees with execution storage")
            costs = []
            for operation in dict.fromkeys(r.operation for r in requests[key]):
                family = tuple(r for r in requests[key] if r.operation == operation)
                cost = task_collective_cost(family, info, source.setup)
                costs.append((operation, cost))
            # The native callback holds one ordered collective slot for the
            # entire reduction (scatter, then optional cross-replica sum).
            phases = (
                TransferPhase(
                    "timed",
                    bytes=sum(cost.transfer_bytes for _, cost in costs),
                    duration_ns=round(
                        sum(cost.duration_seconds for _, cost in costs) * 1e9
                    ),
                    resource="sharded_collectives",
                    submission_ordered=True,
                    timing_class="+".join(f"{op}_{cost.source}" for op, cost in costs),
                ),
            )
            size = record["shard_bytes"]
            fetch = host_phases(info, "fetch", size)
            evict = host_phases(info, "evict", size)
            if key in gradients:
                fetch = (_gradient_zero(source, info, requests[key]), *fetch)
                evict = (*phases, *evict)
            else:
                fetch = (*fetch, *phases)
            alias = replace(alias, transfer=TransferSpec(size, fetch, evict))
        aliases.append(alias)
    return replace(costed, alias_groups=tuple(aliases))


def _gradient_zero(
    program: Program, info: CommunicationInfo, requests: tuple[CollectiveRequest, ...]
) -> TransferPhase:
    """Restore sparse ownership into an otherwise zeroed execution accumulator.

    Calibration records the full-buffer zero separately from reduction timing.
    Without that observation, use the host-fetch rate as an explicit conservative
    bandwidth estimate, never silently model the initialization as free.
    """
    durations, measured = [], True
    for request in requests:
        if request.operation != "reduce_scatter":
            continue
        rows = matching_measurements(
            info,
            {
                **request.to_dict(),
                "setup": program.setup,
                "topology": info.topology,
            },
        )
        duration = next(
            (
                row["zero_duration_seconds"]
                for row in rows
                if row.get("zero_duration_seconds") is not None
            ),
            None,
        )
        measured &= duration is not None
        durations.append(
            duration
            if duration is not None
            else request.tensor_bytes / info.fetch_bytes_per_second
        )
    return TransferPhase(
        "timed",
        duration_ns=round(max(durations) * 1e9),
        resource="gradient_zero",
        timing_class="zero_measurement" if measured else "zero_host_rate_estimate",
    )
