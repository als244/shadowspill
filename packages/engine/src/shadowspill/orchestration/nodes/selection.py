"""Select equivalent assigned GPUs before reserving any process resources."""

from dataclasses import dataclass, replace

from shadowspill_client import Resources
from shadowspill_client._records import integer
from shadowspill_client.protocol.hardware import (
    HardwareDevice,
    HardwareInventory,
    HardwareNode,
    WorkerPlacement,
)

from ..launch.admission import ResourcesUnavailable


class HardwareSelectionError(ValueError):
    """A public placement diagnostic containing no private paths or credentials."""


@dataclass(frozen=True)
class Selection:
    resources: Resources
    workers: tuple[WorkerPlacement, ...]


def select_hardware(
    inventory: HardwareInventory,
    *,
    ranks: int,
    device_bytes: int,
    resources: Resources | None = None,
    wait: bool = False,
) -> Selection:
    integer(ranks, "logical workers", minimum=1)
    integer(device_bytes, "worker device bytes", minimum=1)
    if type(wait) is not bool:
        raise TypeError("resource waiting must be boolean")
    requested = Resources() if resources is None else resources
    nodes = [
        node
        for node in inventory.nodes
        if (requested.pool is None or node.pool == requested.pool)
        and (requested.node is None or node.id == requested.node)
    ]
    if not nodes:
        raise HardwareSelectionError("requested pool/node has no assigned hardware")
    selected: list[tuple[HardwareNode, HardwareDevice]]
    if requested.devices is not None:
        if len(requested.devices) != ranks:
            raise HardwareSelectionError(
                "selected device count differs from logical parallelism"
            )
        selected = []
        for device_id in requested.devices:
            matches = [
                (node, device)
                for node in nodes
                for device in node.devices
                if (
                    device.index == device_id
                    if isinstance(device_id, int)
                    else device.id == device_id
                )
            ]
            if len(matches) != 1:
                raise HardwareSelectionError(
                    "selected device is unavailable or ambiguous in this allocation"
                )
            selected.append(matches[0])
        if len({(node.id, device.id) for node, device in selected}) != ranks:
            raise HardwareSelectionError(
                "device indexes and IDs select the same GPU twice"
            )
        if any(
            device.capacity_bytes < device_bytes
            or (
                requested.gpu_type is not None and device.gpu_type != requested.gpu_type
            )
            for _, device in selected
        ):
            raise HardwareSelectionError(
                "selected devices cannot satisfy the requested type/budget"
            )
    else:
        groups: dict[
            tuple[str, str, str | None, int], list[tuple[HardwareNode, HardwareDevice]]
        ] = {}
        for node in nodes:
            for device in node.devices:
                if device.capacity_bytes < device_bytes or (
                    requested.gpu_type is not None
                    and device.gpu_type != requested.gpu_type
                ):
                    continue
                # Unknown hardware cannot be assumed equivalent across nodes.
                key = (
                    node.pool,
                    device.gpu_type or node.id,
                    node.driver,
                    device.capacity_bytes,
                )
                groups.setdefault(key, []).append((node, device))
        choices = []
        for candidates in groups.values():
            if len(candidates) < ranks:
                continue
            # Prefer a complete node to avoid introducing cross-node communication
            # when the same GPU class is available together. Consider all nodes
            # as well when no single node supplies enough available GPUs.
            subsets = [candidates] + [
                [row for row in candidates if row[0].id == node.id] for node in nodes
            ]
            for subset in subsets:
                if len(subset) >= ranks:
                    ordered = sorted(
                        subset,
                        key=lambda row: (not row[1].available, row[0].id, row[1].index),
                    )[:ranks]
                    choices.append(ordered)
        if not choices:
            raise HardwareSelectionError(
                "logical worker count/budget cannot fit equivalent assigned GPUs"
            )
        selected = min(
            choices,
            key=lambda rows: (
                sum(not device.available for _, device in rows),
                len({node.id for node, _ in rows}),
                sum(device.capacity_bytes for _, device in rows),
                tuple((node.id, device.id) for node, device in rows),
            ),
        )
    if not wait and any(not device.available for _, device in selected):
        raise ResourcesUnavailable("compatible assigned workers have live owners")
    if len({node.pool for node, _ in selected}) != 1:
        raise HardwareSelectionError("selected devices span different compute pools")
    # Explicit heterogeneous selection is expressed with WorkerSettings; this
    # homogeneous selector must never silently group different GPU classes.
    classes = {
        (device.gpu_type or node.id, device.capacity_bytes, node.driver)
        for node, device in selected
    }
    if len(classes) != 1:
        raise HardwareSelectionError(
            "mixed GPU types require explicit per-worker settings"
        )
    node, device = selected[0]
    normalized = replace(
        requested,
        pool=node.pool,
        gpu_type=device.gpu_type,
        devices=None
        if requested.devices is None
        else tuple(device.id for _, device in selected),
    )
    return Selection(
        normalized,
        tuple(
            WorkerPlacement(rank, node.id, device)
            for rank, (node, device) in enumerate(selected)
        ),
    )
