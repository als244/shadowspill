"""Publish already observed node capacity and current exclusive resource claims."""

from shadowspill_client.protocol.hardware import HardwareDevice, HardwareNode

from ..launch.admission import NodeAdmission


def node_hardware(node: NodeAdmission, *, pool: str) -> HardwareNode:
    capacity, used = node.capacity, node.used
    return HardwareNode(
        capacity.node_id,
        pool,
        tuple(
            HardwareDevice(
                device.id,
                index,
                device.name,
                device.capacity_bytes,
                device.id not in used.devices,
            )
            for index, device in enumerate(capacity.devices)
        ),
        capacity.host_bytes,
        capacity.host_bytes - used.host_bytes,
        capacity.cpu_count,
        capacity.driver,
    )
