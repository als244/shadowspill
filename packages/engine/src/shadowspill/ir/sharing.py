"""Derived physical footprint of runtime-global shared alias groups."""

from __future__ import annotations

from dataclasses import dataclass

from .program import ShadowSpillProgram, SharedResidencyPolicy


@dataclass(frozen=True, slots=True)
class SharedResidencyFootprint:
    """Memory a program's shared alias groups hold outside any schedule, in bytes.

    `shared_residency_footprint` returns it. Planning subtracts a device's
    shared bytes from its capacity before a search.

    Attributes
    ----------
    device_bytes : tuple of (str, int)
        Each device's identity and the size of the shared groups on it, for
        every device of the program in its order; zero for a device with none.
    spill_bytes : int
        Spill-copy size of the shared groups that retain their spill copy.
    alias_group_ids : tuple of str
        The shared groups, in program order.
    """

    device_bytes: tuple[tuple[str, int], ...]
    spill_bytes: int
    alias_group_ids: tuple[str, ...]

    def for_device(self, device_id: str) -> int:
        """Return the bytes of shared groups on one device.

        Parameters
        ----------
        device_id : str
            One of the program's devices.

        Returns
        -------
        int
            The device's entry in ``device_bytes``.

        Raises
        ------
        KeyError
            If the program has no such device.
        """

        for candidate, size_bytes in self.device_bytes:
            if candidate == device_id:
                return size_bytes
        raise KeyError(device_id)


def shared_residency_footprint(program: ShadowSpillProgram) -> SharedResidencyFootprint:
    """Sum the memory a program's shared alias groups hold.

    A group with a `SharedResidencyPolicy` counts its size on its device, and
    its spill-copy size when it retains that copy.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program.

    Returns
    -------
    SharedResidencyFootprint
        The bytes per device and in spill, and the groups counted.
    """

    by_device = {device.device_id: 0 for device in program.devices}
    spill_bytes = 0
    aliases: list[str] = []
    for alias in program.alias_groups:
        if not isinstance(alias.shared_residency, SharedResidencyPolicy):
            continue
        by_device[alias.device_id] += alias.size_bytes
        if alias.retain_spill_copy:
            spill_bytes += alias.spill_size_bytes
        aliases.append(alias.alias_group_id)
    return SharedResidencyFootprint(
        tuple(by_device.items()),
        spill_bytes,
        tuple(aliases),
    )


__all__ = ["SharedResidencyFootprint", "shared_residency_footprint"]
