"""Optional deployment selection, separate from portable training settings."""

from collections.abc import Sequence
from dataclasses import dataclass

from .._records import Record, integer, text


@dataclass(frozen=True)
class Resources(Record):
    """Select assigned hardware; omission permits equivalent compatible devices.

    GPU names and node/device identities come from ``list_hardware``. Integer
    devices are indexes within an explicitly named node's assigned inventory;
    stable device IDs may be used instead. A selector is not a reservation.
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
