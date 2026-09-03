"""Physical replay, budget reconciliation, and runtime sealing."""

from __future__ import annotations

import ctypes

from shadowspill.errors import AdmissionError
from shadowspill.ir import ExecutionPlan, MemoryActionKind, PhysicalAdmission
from shadowspill.planner.admission.layout import FixedPhysicalLayout
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.allocator import InstalledAllocator

from ...runtime_adapter import PlanMemory
from ..common import round_up

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
    """Return one safe lower bound for all sealed runtime record tables."""

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


def reconcile_spill_pool(*, predicted_peak: int, budget: int) -> None:
    """Reject a selected schedule that exceeds its public spill budget."""

    if predicted_peak < 0:
        raise AdmissionError("predicted spill peak must be non-negative")
    if predicted_peak > budget:
        raise AdmissionError(
            "predicted spill-pool peak exceeds the selected plan budget: "
            f"peak={predicted_peak}, budget={budget}"
        )


def physical_admission(
    memory: PlanMemory,
    installed: InstalledAllocator,
    *,
    workspace_reserve: int,
    predicted_spill_peak_bytes: int,
    predicted_fragmentation_bytes: int,
) -> PhysicalAdmission:
    """Describe the callable's admitted resources after spatial replay.

    The runtime may own a larger spill pool than this plan is allowed to use.
    ``spill_reservation_bytes`` therefore records this callable's predicted
    spill peak, not the process-lifetime pool allocation.
    """

    reconcile_spill_pool(
        predicted_peak=predicted_spill_peak_bytes,
        budget=memory.spill_budget,
    )

    return PhysicalAdmission(
        device_budget_bytes=(
            memory.execution.physical_capacity or memory.execution_budget
        ),
        spill_budget_bytes=memory.spill_budget,
        baseline_bytes=int(installed.admission.baseline_bytes),
        provider_headroom_bytes=int(installed.admission.provider_headroom_bytes),
        slab_bytes=memory.execution_budget,
        workspace_reserve_bytes=workspace_reserve,
        spill_reservation_bytes=predicted_spill_peak_bytes,
        predicted_fragmentation_bytes=predicted_fragmentation_bytes,
    )


def seal_physical_budget(
    installed: InstalledAllocator,
    execution_plan: ExecutionPlan,
    fixed_layout: FixedPhysicalLayout,
) -> None:
    """Seal provider headroom and complete steady-state record inventories."""

    if fixed_layout.program_digest != execution_plan.program.digest:
        raise AdmissionError("fixed layout belongs to a different Program")
    if fixed_layout.schedule_digest != execution_plan.schedule.digest:
        raise AdmissionError("fixed layout belongs to a different memory schedule")

    library = installed.library
    status = int(library.shadowspill_pytorch_check_physical_budget())
    if status != 0:
        raise AdmissionError(
            f"provider allocations exceeded physical admission (status {status})"
        )
    statistics = AdapterStatistics()
    status = int(
        library.shadowspill_pytorch_allocator_statistics(ctypes.byref(statistics))
    )
    if status != 0:
        raise AdmissionError(f"allocator statistics failed with status {status}")
    required = max(
        int(installed.admission.provider_headroom_bytes),
        round_up(
            int(statistics.observed_external_high_water_bytes) + 64 * _MIB,
            64 * _MIB,
        ),
    )
    initial_transfers = sum(
        item.location.value == "device"
        for item in execution_plan.schedule.initial_residency
    )
    scheduled_transfers = sum(
        item.kind in {MemoryActionKind.EVICT, MemoryActionKind.FETCH}
        for item in execution_plan.schedule.actions
    )
    # The host dispatcher may publish a complete step before CUDA retires its
    # earliest task allocations and transfer reservations.  Reserve metadata
    # for every lifetime certified by the physical layout; task/transfer
    # counts alone undercount allocation-heavy compiled graphs.  The C seal
    # uses one common lower bound for event, retirement, and per-pool lease
    # records, so the larger of the event and lifetime inventories is safe for
    # every component without adding a second sizing contract.
    runtime_record_reserve = _runtime_record_reserve(
        selected_task_count=len(
            execution_plan.program.selected_tasks(execution_plan.selections)
        ),
        initial_transfer_count=initial_transfers,
        scheduled_transfer_count=scheduled_transfers,
        event_pool_peak_in_use=int(statistics.runtime.event_lease_peak_in_use),
        fixed_lifetime_count=len(fixed_layout.placements),
        dynamic_lifetime_count=len(fixed_layout.dynamic_lifetimes),
    )
    status = int(
        library.shadowspill_pytorch_seal_physical_budget(
            required,
            runtime_record_reserve,
        )
    )
    if status != 0:
        reserved = int(installed.admission.provider_headroom_bytes)
        raise AdmissionError(
            "observed provider memory exceeds the reserved headroom: "
            f"required={required}, reserved={reserved}"
        )


__all__ = ["physical_admission", "reconcile_spill_pool", "seal_physical_budget"]
