"""Optional deployment selection, separate from portable training settings."""

from collections.abc import Sequence
from dataclasses import dataclass

from .._records import Record, integer, text


@dataclass(frozen=True)
class Resources(Record):
    """Hardware a run's workers may use.

    Every field is optional; what is left out lets the service choose
    equivalent compatible devices. Names and device IDs come from
    `ServiceClient.list_hardware`. A selection reserves nothing.

    Parameters
    ----------
    pool : str, optional
        Compute pool, as the service operator names it.
    node : str, optional
        Node name. Required when ``devices`` holds indexes.
    gpu_type : str, optional
        GPU type, such as ``"NVIDIA GeForce RTX 5090"``. Any matching devices
        may be chosen.
    devices : sequence of str or int, optional
        Exact devices, one per worker in rank order: stable device IDs, or
        indexes into ``node``'s assigned inventory. Between 1 and 1024
        entries, without duplicates.

    Raises
    ------
    ValueError
        If a name or ID is longer than 256 characters, ``devices`` is empty
        or repeats a device, or it holds indexes without ``node``.
    """

    pool: str | None = None
    node: str | None = None
    gpu_type: str | None = None
    devices: Sequence[str | int] | None = None

    def __post_init__(self) -> None:
        for name in ("pool", "node", "gpu_type"):
            value = getattr(self, name)
            if value is not None and len(text(value, name)) > 256:
                raise ValueError(f"{name} exceeds its size bound")
        if self.devices is not None:
            if (
                not isinstance(self.devices, (tuple, list))
                or not 1 <= len(self.devices) <= 1024
            ):
                raise ValueError("devices require a nonempty bounded sequence")
            devices = tuple(self.devices)
            for value in devices:
                if isinstance(value, int):
                    integer(value, "assigned device index")
                    if self.node is None:
                        raise ValueError("device indexes require an explicit node")
                elif len(text(value, "assigned device ID")) > 256:
                    raise ValueError("assigned device ID exceeds its size bound")
            if len(set(devices)) != len(devices):
                raise ValueError("device selection contains duplicates")
            object.__setattr__(self, "devices", devices)
