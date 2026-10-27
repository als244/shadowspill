"""Worker setup consumes a physical Plan; it never builds or calibrates."""

import torch

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.plan import Plan
from shadowspill.memory import DevicePool, PinnedHostPool, SpillPool, TransferRoute
from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.runtime import Runtime
from shadowspill.runtime.plan import RuntimeBridge
from shadowspill.runtime.plan.lifecycle import PlanMemory, begin_plan


def open_worker(
    plan: Plan,
    device: int,
    external_headroom_bytes: int,
    spill: SpillPool | None = None,
) -> Runtime:
    if plan.simulation_only_reasons:
        raise ValueError(
            "Plan is simulation-only: " + "; ".join(plan.simulation_only_reasons)
        )
    if len(plan.program.devices) != 1:
        raise NotImplementedError("Executor requires one symmetric rank-local device")
    if plan.program.setup["torch"] != torch.__version__:
        raise ValueError("Plan requires a different PyTorch build")
    require_compiled_device(plan.program, device)
    spill = spill or PinnedHostPool(plan.memory_budgets.spill_bytes)
    if not isinstance(spill, SpillPool):
        raise TypeError("spill must be a SpillPool configuration")
    if spill.capacity != plan.memory_budgets.spill_bytes:
        raise ValueError("spill capacity must equal the Plan's spill_bytes budget")
    measured_kind = plan.communication_info.metadata.get("spill_kind")
    if measured_kind is not None and measured_kind != spill.kind_name:
        raise ValueError(
            "Plan communication was measured for a different spill backend; "
            "replan with its CommunicationInfo"
        )
    if getattr(spill, "image", None) is not None:
        raise ValueError(
            "Executor does not yet adopt SSD pool images; use a tensor loader"
        )
    runtime = Runtime(
        frontend=PyTorchFrontend(),
        pools={
            "execution": DevicePool(
                plan.memory_budgets.execution_bytes,
                device=device,
                external_headroom=external_headroom_bytes,
                reject_overbudget=True,
            ),
            "spill": spill,
        },
        routes={
            "fetch": TransferRoute("spill", "execution"),
            "evict": TransferRoute("execution", "spill"),
        },
        calibrate=False,
    )
    try:
        if build_setup(torch.device("cuda", device)) != plain(plan.program.setup):
            raise ValueError("Plan build/target identity differs from this worker")
        return runtime
    except BaseException:
        runtime.close()
        raise


def open_path(
    runtime: Runtime,
    plan: Plan,
    device: int,
    *,
    slab_owner: int | None = None,
) -> tuple[PlanMemory, RuntimeBridge]:
    memory = begin_plan(
        runtime,
        execution="execution",
        spill="spill",
        execution_device=device,
    )
    bridge = RuntimeBridge(
        runtime,
        plan._costed_program,
        memory.plan_handle,
        execution_pool_id=memory.execution.pool_id,
        spill_pool_id=memory.spill.pool_id,
        slab_owner=slab_owner,
    )
    return memory, bridge
