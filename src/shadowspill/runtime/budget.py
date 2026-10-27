"""Check physical GPU headroom and seal runtime record reservations."""

from __future__ import annotations

import ctypes

from shadowspill.errors import AdmissionError
from shadowspill.runtime.abi import AdapterStatistics, PhysicalMemory
from shadowspill.runtime.bootstrap import InstalledRuntime

_MIB = 1 << 20


def _runtime_record_reserve(
    *,
    selected_task_count: int,
    initial_transfer_count: int,
    scheduled_transfer_count: int,
    event_pool_peak_in_use: int,
    fixed_lifetime_count: int,
    dynamic_lifetime_count: int,
) -> int:
    """Return the size every runtime record table starts at.

    A bound for the inventories the plan bounds -- event leases and
    retirement records -- and a warm start for the ones it does not. How
    many lease records a pool needs is not derivable here: one is taken per
    live lease and one more each time a free range is split to fit a
    request, so the peak follows the order allocations and releases happen
    in and how far the dispatcher runs ahead of the device. A pool grows its
    own metadata past this rather than refusing a request its bytes were
    there to serve, so what this has to be is large enough for the steady
    state, not large enough for every future model.
    """

    event_records = max(
        256,
        selected_task_count
        + initial_transfer_count
        + scheduled_transfer_count
        + 2 * event_pool_peak_in_use
        + 64,
    )
    lifetime_records = fixed_lifetime_count + dynamic_lifetime_count + 64
    return max(event_records, lifetime_records)


def check_physical_budget(installed: InstalledRuntime) -> None:
    """Check the same reservation at admission and after distributed updates."""
    library = installed.library
    status = int(library.shadowspill_pytorch_check_physical_budget())
    if status != 0:
        physical = PhysicalMemory()
        details = ""
        if library.shadowspill_pytorch_physical_memory(ctypes.byref(physical)) == 0:
            admission = installed.admission
            external = max(
                0,
                int(physical.process_bytes)
                - int(admission.baseline_bytes)
                - int(admission.allocator_pool_bytes),
            )
            details = (
                f": process={int(physical.process_bytes)}, "
                f"budget={int(admission.device_budget_bytes)}, "
                f"external={external}, "
                f"external_headroom={int(admission.external_headroom_bytes)}, "
                f"baseline={int(admission.baseline_bytes)}, "
                f"allocator_pool={int(admission.allocator_pool_bytes)}"
            )
        raise AdmissionError(f"physical budget check failed (status {status}){details}")


def seal_worker_budget(
    installed: InstalledRuntime,
    *,
    selected_task_count: int,
    initial_transfer_count: int,
    scheduled_transfer_count: int,
    fixed_lifetime_count: int,
    dynamic_lifetime_count: int,
) -> None:
    """Seal the admitted worker's physical budget and reusable record inventory."""
    check_physical_budget(installed)
    library = installed.library
    statistics = AdapterStatistics()
    status = int(
        library.shadowspill_pytorch_allocator_statistics(ctypes.byref(statistics))
    )
    if status != 0:
        raise AdmissionError(f"allocator statistics failed with status {status}")
    alignment = 64 * _MIB
    observed = int(statistics.observed_external_high_water_bytes) + alignment
    required = max(
        int(installed.admission.external_headroom_bytes),
        ((observed + alignment - 1) // alignment) * alignment,
    )
    runtime_record_reserve = _runtime_record_reserve(
        selected_task_count=selected_task_count,
        initial_transfer_count=initial_transfer_count,
        scheduled_transfer_count=scheduled_transfer_count,
        event_pool_peak_in_use=int(statistics.runtime.event_lease_peak_in_use),
        fixed_lifetime_count=fixed_lifetime_count,
        dynamic_lifetime_count=dynamic_lifetime_count,
    )
    status = int(
        library.shadowspill_pytorch_seal_physical_budget(
            required,
            runtime_record_reserve,
        )
    )
    if status != 0:
        reserved = int(installed.admission.external_headroom_bytes)
        raise AdmissionError(
            "observed external memory exceeds the reserved headroom: "
            f"required={required}, reserved={reserved}, "
            f"observed_external={int(statistics.observed_external_high_water_bytes)}"
        )
