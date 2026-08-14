"""Physical replay, budget reconciliation, and runtime sealing."""

from __future__ import annotations

import ctypes

from shadowspill.ir import ExecutionPlan, MemoryActionKind, PhysicalAdmission
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.allocator import InstalledAllocator
from shadowspill.runtime import SlabReplay

from ...contracts import AdmissionError
from ...runtime_adapter import PlanMemory
from ..common import round_up

_MIB = 1 << 20


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
    slab_replay: SlabReplay,
) -> PhysicalAdmission:
    """Describe the physical resources admitted after spatial replay."""

    return PhysicalAdmission(
        device_budget_bytes=(
            memory.execution.physical_capacity or memory.execution_budget
        ),
        host_budget_bytes=memory.spill_budget,
        context_bytes=int(installed.admission.context_bytes),
        provider_headroom_bytes=int(installed.admission.provider_headroom_bytes),
        slab_bytes=memory.execution_budget,
        workspace_reserve_bytes=workspace_reserve,
        host_reservation_bytes=int(installed.admission.spill_pool_bytes),
        predicted_fragmentation_bytes=slab_replay.peak_fragmentation_bytes,
    )


def seal_physical_budget(
    installed: InstalledAllocator,
    execution_plan: ExecutionPlan,
) -> None:
    """Seal provider headroom and the complete steady-state event inventory."""

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
        item.kind in {MemoryActionKind.OFFLOAD, MemoryActionKind.PREFETCH}
        for item in execution_plan.schedule.actions
    )
    event_pool_reserve = max(
        256,
        len(execution_plan.program.selected_tasks(execution_plan.selections))
        + initial_transfers
        + scheduled_transfers
        + 2 * int(statistics.cuda.event_pool_peak_in_use)
        + 64,
    )
    status = int(
        library.shadowspill_pytorch_seal_physical_budget(required, event_pool_reserve)
    )
    if status != 0:
        reserved = int(installed.admission.provider_headroom_bytes)
        raise AdmissionError(
            "observed provider memory exceeds the reserved headroom: "
            f"required={required}, reserved={reserved}"
        )


__all__ = ["physical_admission", "reconcile_spill_pool", "seal_physical_budget"]
