"""Inspectable assigned hardware and placement snapshots, without GPU dependencies."""

from collections.abc import Sequence
from dataclasses import dataclass

from .._records import Record, integer, record, text


def _name(value: str, name: str) -> None:
    if len(text(value, name)) > 256:
        raise ValueError(f"{name} exceeds its size bound")


@dataclass(frozen=True)
class HardwareDevice(Record):
    """A device assigned to a node.

    Parameters
    ----------
    id : str
        Stable device ID, accepted by `Resources` ``devices``.
    index : int
        Position in the node's assigned devices, from zero, accepted by
        `Resources` ``devices`` together with ``node``.
    gpu_type : str or None
        GPU name the device reports, accepted by `Resources` ``gpu_type``.
        None when unknown.
    capacity_bytes : int
        Device memory, in bytes.
    available : bool
        True unless one of the service's runs holds the device.
    """

    id: str
    index: int
    gpu_type: str | None
    capacity_bytes: int
    available: bool

    def __post_init__(self) -> None:
        _name(self.id, "device ID")
        integer(self.index, "assigned device index")
        if self.gpu_type is not None:
            _name(self.gpu_type, "GPU type")
        integer(self.capacity_bytes, "device capacity", minimum=1)
        if type(self.available) is not bool:
            raise TypeError("device availability must be boolean")


@dataclass(frozen=True)
class HardwareNode(Record):
    """A machine the service can assign, with its devices and host memory.

    Parameters
    ----------
    id : str
        Node ID, accepted by `Resources` ``node``.
    pool : str
        Compute pool the node belongs to, accepted by `Resources` ``pool``.
    devices : sequence of HardwareDevice
        The node's assigned devices, in ``index`` order.
    host_capacity_bytes : int
        Host memory assigned to the node, in bytes.
    host_available_bytes : int
        Part of ``host_capacity_bytes`` the service has not claimed, in bytes.
    cpu_count : int, optional
        Number of CPUs assigned to the node. None when unknown.
    driver : str, optional
        GPU driver version. None when unknown.
    """

    id: str
    pool: str
    devices: Sequence[HardwareDevice]
    host_capacity_bytes: int
    host_available_bytes: int
    cpu_count: int | None = None
    driver: str | None = None

    def __post_init__(self) -> None:
        _name(self.id, "node ID")
        _name(self.pool, "pool")
        if self.driver is not None:
            _name(self.driver, "GPU driver")
        if self.cpu_count is not None:
            integer(self.cpu_count, "assigned CPUs", minimum=1)
        integer(self.host_capacity_bytes, "assigned host capacity")
        integer(self.host_available_bytes, "available host capacity")
        if self.host_available_bytes > self.host_capacity_bytes:
            raise ValueError("available host capacity exceeds its assignment")
        if not isinstance(self.devices, (list, tuple)) or len(self.devices) > 1024:
            raise ValueError("assigned device list exceeds its bound")
        devices = tuple(record(HardwareDevice, item, "device") for item in self.devices)
        if len({item.id for item in devices}) != len(devices) or tuple(
            item.index for item in devices
        ) != tuple(range(len(devices))):
            raise ValueError("devices require unique IDs and ordered assigned indexes")
        object.__setattr__(self, "devices", devices)


@dataclass(frozen=True)
class HardwareInventory(Record):
    """The hardware a service can assign, from `ServiceClient.list_hardware`.

    Parameters
    ----------
    nodes : sequence of HardwareNode
        The nodes, each with a distinct ID.
    """

    nodes: Sequence[HardwareNode]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, (list, tuple)) or len(self.nodes) > 1024:
            raise ValueError("node inventory exceeds its bound")
        nodes = tuple(record(HardwareNode, item, "node") for item in self.nodes)
        if len({item.id for item in nodes}) != len(nodes):
            raise ValueError("node inventory contains duplicate IDs")
        if sum(len(node.devices) for node in nodes) > 1024:
            raise ValueError("assigned inventory exceeds its device bound")
        object.__setattr__(self, "nodes", nodes)

    def summary(self) -> str:
        """Describe the inventory as a table for printing.

        Returns
        -------
        str
            One row per device with its pool, node, index, GPU, memory in GiB
            and status, then one line per node with its CPUs, host memory and
            driver.
        """
        rows = [("Pool", "Node", "Device", "GPU", "GiB", "Status")]
        for node in self.nodes:
            rows.extend(
                (
                    node.pool,
                    node.id,
                    str(device.index),
                    device.gpu_type or "Unknown",
                    f"{device.capacity_bytes / (1 << 30):.1f}",
                    "Available" if device.available else "Busy",
                )
                for device in node.devices
            )
        widths = [max(len(row[column]) for row in rows) for column in range(6)]
        lines = [
            "  ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            ).rstrip()
            for row in rows
        ]
        lines.append("")
        for node in self.nodes:
            cpus = node.cpu_count if node.cpu_count is not None else "unknown"
            lines.append(
                f"{node.id}: CPUs {cpus}; "
                f"host {node.host_available_bytes / (1 << 30):.1f} / "
                f"{node.host_capacity_bytes / (1 << 30):.1f} GiB available / assigned; "
                f"driver {node.driver or 'unknown'}."
            )
        lines.append(
            "Availability reflects service claims; inspection reserves no hardware."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class WorkerPlacement(Record):
    """The device proposed for one worker.

    Parameters
    ----------
    rank : int
        The worker's rank.
    node : str
        ID of the node holding the device.
    device : HardwareDevice
        The device.
    """

    rank: int
    node: str
    device: HardwareDevice

    def __post_init__(self) -> None:
        integer(self.rank, "logical rank")
        _name(self.node, "node ID")
        object.__setattr__(
            self, "device", record(HardwareDevice, self.device, "device")
        )
