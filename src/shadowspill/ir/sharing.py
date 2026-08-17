"""Derived physical footprint of runtime-global shared alias groups."""

from __future__ import annotations

from dataclasses import dataclass

from .program import Program, SharedResidencyPolicy


@dataclass(frozen=True, slots=True)
class SharedResidencyFootprint:
    """Bytes charged once outside any callable-owned movable schedule."""

    device_bytes: tuple[tuple[str, int], ...]
    spill_bytes: int
    alias_group_ids: tuple[str, ...]

    def for_device(self, device_id: str) -> int:
        """Return shared execution-resident bytes for one Program device."""

        for candidate, size_bytes in self.device_bytes:
            if candidate == device_id:
                return size_bytes
        raise KeyError(device_id)


def shared_residency_footprint(program: Program) -> SharedResidencyFootprint:
    """Summarize shared leases without changing semantic object sizes."""

    by_device = {device.device_id: 0 for device in program.devices}
    spill_bytes = 0
    aliases: list[str] = []
    for alias in program.alias_groups:
        if not isinstance(alias.shared_residency, SharedResidencyPolicy):
            continue
        by_device[alias.device_id] += alias.size_bytes
        if alias.retain_spill_copy:
            spill_bytes += alias.size_bytes
        aliases.append(alias.alias_group_id)
    return SharedResidencyFootprint(
        tuple(by_device.items()),
        spill_bytes,
        tuple(aliases),
    )


__all__ = ["SharedResidencyFootprint", "shared_residency_footprint"]
