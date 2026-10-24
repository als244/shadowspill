"""Materialize transfer costs from saved geometry, without task remeasurement."""

from dataclasses import replace

from shadowspill.artifacts.program import Program
from shadowspill.communication import CommunicationInfo
from shadowspill.communication.collectives import collective_cost
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
    shards = source.bindings.get("weight_shards", {})
    aliases = []
    for alias in costed.alias_groups:
        record = shards.get(alias.alias_group_id)
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
                raise ValueError(
                    "weight shard geometry disagrees with execution storage"
                )
            # One symmetric schedule is conservatively timed by the slowest
            # group instance. Costs are per rank, not multiplied by world size.
            costs = [
                collective_cost(
                    info,
                    operation="all_gather",
                    ranks=tuple(group),
                    tensor_bytes=record["padded_bytes"],
                    dtype=record["dtype"],
                    setup=source.setup,
                    inplace=True,
                )
                for group in record["groups"]
            ]
            cost = max(costs, key=lambda value: value.duration_seconds)
            copy_bytes = record["shard_bytes"]
            alias = replace(
                alias,
                transfer=TransferSpec(
                    copy_bytes,
                    (
                        *host_phases(info, "fetch", copy_bytes),
                        TransferPhase(
                            "timed",
                            bytes=cost.transfer_bytes,
                            duration_ns=round(cost.duration_seconds * 1e9),
                            resource="sharded_collectives",
                            submission_ordered=True,
                            timing_class=f"all_gather_{cost.source}",
                        ),
                    ),
                    host_phases(info, "evict", copy_bytes),
                ),
            )
        aliases.append(alias)
    return replace(costed, alias_groups=tuple(aliases))
