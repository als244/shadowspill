"""Size a run's automatic memory budgets from the hardware its workers use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from shadowspill_client.config import Memory
from shadowspill_client.protocol.hardware import HardwareInventory, WorkerPlacement

#: Part of a worker's share of a memory that an automatic budget takes.
AUTOMATIC_SHARE = 0.9


def automatic(memory: Memory) -> bool:
    """Whether either budget is automatic (zero)."""
    return memory.device_gib == 0 or memory.spill_gib == 0


def device_request_bytes(memory: Memory) -> int:
    """Device memory, in bytes, a worker's GPU must have; any for automatic."""
    return 1 if memory.device_gib == 0 else int(memory.device_gib * (1 << 30))


def resolve_memory(
    memory: Memory,
    workers: Sequence[WorkerPlacement],
    inventory: HardwareInventory,
    reserved_host_bytes: Mapping[str, int] | None = None,
) -> Memory:
    """Replace each automatic budget with its size on the selected workers.

    A worker's device share is its GPU's memory; its spill share is its node's
    host memory, less what ``reserved_host_bytes`` holds there for the run
    beside the spill pools (by node ID), divided evenly among the node's GPUs.
    One budget applies to every worker, so each automatic budget takes the
    smallest worker's share.
    """
    if not automatic(memory):
        return memory
    if not workers:
        raise ValueError("automatic memory budgets require selected workers")
    nodes = {node.id: node for node in inventory.nodes}
    device = memory.device_gib or _share(
        min(worker.device.capacity_bytes for worker in workers), "device"
    )
    spill = memory.spill_gib or _share(
        min(
            max(
                0,
                nodes[worker.node].host_capacity_bytes
                - (reserved_host_bytes or {}).get(worker.node, 0),
            )
            // len(nodes[worker.node].devices)
            for worker in workers
        ),
        "spill",
    )
    return Memory(device_gib=device, spill_gib=spill)


def _share(capacity_bytes: int, name: str) -> float:
    # Whole MiB, so the size reads back exactly in GiB.
    mebibytes = int(capacity_bytes * AUTOMATIC_SHARE) >> 20
    if mebibytes < 1:
        raise ValueError(f"the selected workers have no memory for a {name} budget")
    return mebibytes / 1024
