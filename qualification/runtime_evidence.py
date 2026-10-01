"""Small helpers for reading allocator and runtime qualification evidence."""

from __future__ import annotations

import ctypes
from typing import Any

from shadowspill.runtime.abi import AdapterStatistics
from shadowspill.runtime.bootstrap import installed_runtime


def measured_rate_clause(runtime: Any) -> str:
    """What this runtime's routes measured, solo and concurrent, for a report.

    **Both, and labelled**, because they are different measurements and the one
    planning uses is the concurrent one. Calibration measures each route alone,
    then measures both at once and overwrites `bandwidth_bytes_per_second` with
    that second number -- so an unlabelled figure reads as the link's rate,
    invites the conclusion that the link has degraded, and is not comparable to
    a single-direction benchmark. The solo figure is what `ib_read_bw` and
    `ib_write_bw` are comparable to.

    Empty where the routes cannot be read: reporting must never be what fails a
    matrix cell.
    """

    try:
        capabilities = runtime.transfer_capabilities
        fetch = capabilities.route("spill", "execution")
        evict = capabilities.route("execution", "spill")
    except Exception:
        return ""
    clause = (
        "; measured solo: fetch "
        f"{fetch.solo_bandwidth_bytes_per_second / 1e9:.1f} GB/s, evict "
        f"{evict.solo_bandwidth_bytes_per_second / 1e9:.1f} GB/s"
    )
    # Zero, or the 1 a failed measurement floors to, means the bidirectional
    # pass did not run and there is nothing to contrast against.
    if (
        min(
            fetch.concurrent_bandwidth_bytes_per_second,
            evict.concurrent_bandwidth_bytes_per_second,
        )
        > 1
    ):
        clause += (
            "; concurrent: fetch "
            f"{fetch.concurrent_bandwidth_bytes_per_second / 1e9:.1f} GB/s, evict "
            f"{evict.concurrent_bandwidth_bytes_per_second / 1e9:.1f} GB/s"
        )
    return clause


def adapter_statistics() -> AdapterStatistics:
    """Return one consistent snapshot from the installed PyTorch adapter."""

    installed = installed_runtime()
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

    installed = installed_runtime()
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
