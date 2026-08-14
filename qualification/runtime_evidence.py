"""Small helpers for reading native allocator/runtime qualification evidence."""

from __future__ import annotations

import ctypes
from typing import Any

from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.allocator import installed_allocator


def adapter_statistics() -> AdapterStatistics:
    """Return one consistent snapshot from the installed PyTorch adapter."""

    installed = installed_allocator()
    if installed is None:
        raise RuntimeError("ShadowSpill allocator is not installed")
    result = AdapterStatistics()
    status = int(
        installed.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result))
    )
    if status != 0:
        raise RuntimeError(f"allocator statistics failed with status {status}")
    return result


def check_physical_budget() -> int:
    """Return zero only when current physical use remains within admission."""

    installed = installed_allocator()
    if installed is None:
        raise RuntimeError("ShadowSpill allocator is not installed")
    return int(installed.library.shadowspill_pytorch_check_physical_budget())


def statistics_dict(value: AdapterStatistics) -> dict[str, Any]:
    """Convert the nested ctypes statistics record into plain integers."""

    return _structure_dict(value)


def _structure_dict(value: ctypes.Structure) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in value._fields_:
        name = field[0]
        item = getattr(value, name)
        result[name] = (
            _structure_dict(item) if isinstance(item, ctypes.Structure) else int(item)
        )
    return result


__all__ = ["adapter_statistics", "check_physical_budget", "statistics_dict"]
