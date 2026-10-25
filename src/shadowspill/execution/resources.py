"""Shared stream setup and physical reservations for admitted invocation paths."""

from collections.abc import Mapping, Sequence

import torch

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.plan import Plan
from shadowspill.ir import MemoryActionKind
from shadowspill.pipeline.admission.physical import seal_worker_budget
from shadowspill.pytorch.resource_streams import ResourceStreams
from shadowspill.runtime.bootstrap import installed_runtime


def worker_headroom(plans: Sequence[Plan], requested: int | None) -> int:
    reservations = [
        p.profiled_program.metadata.get("external_headroom_bytes") for p in plans
    ]
    reserved = max(value if value is not None else 512 << 20 for value in reservations)
    if requested is None:
        requested = reserved
    elif all(value is not None for value in reservations) and requested > reserved:
        raise ValueError(
            "Executor external_headroom_bytes exceeds the Plan reservation; "
            "replan with profiles that include the requested headroom"
        )
    if (
        plans[0].program.parallelism is not None
        and plans[0].program.parallelism.world_size > 1
    ):
        calibrations = {
            digest(p.communication_info.to_dict()): p.communication_info for p in plans
        }
        requested += sum(info.fixed_device_bytes or 0 for info in calibrations.values())
    return requested


def plan_inventory(plan: Plan, additional: Mapping[str, Plan]) -> dict[str, Plan]:
    plans = {"default": plan, **{f"named.{name}": p for name, p in additional.items()}}
    if plan._accumulation_plan is not None:
        plans["accumulate"] = plan._accumulation_plan
    return plans


def worker_streams(plans: Sequence[Plan], device: int) -> ResourceStreams:
    selected = [
        (task, plan)
        for plan in plans
        for task in plan._costed_program.selected_tasks(plan.selections)
    ]
    streams = ResourceStreams(
        torch.device("cuda", device), (task.resource for task, _ in selected)
    )
    streams.initialize_blas(
        task.resource
        for task, plan in selected
        if task.profile_id in plan.program.executables
    )
    return streams


def seal_paths(plans: Sequence[Plan]) -> None:
    installed = installed_runtime()
    assert installed is not None
    layouts = [plan.memory_layout for plan in plans]
    assert all(layout is not None for layout in layouts)
    seal_worker_budget(
        installed,
        selected_task_count=sum(
            len(p._costed_program.selected_tasks(p.selections)) for p in plans
        ),
        initial_transfer_count=sum(
            r.location.value == "device"
            for p in plans
            for r in p.schedule.initial_residency
        ),
        scheduled_transfer_count=sum(
            a.kind in {MemoryActionKind.FETCH, MemoryActionKind.EVICT}
            for p in plans
            for a in p.schedule.actions
        ),
        fixed_lifetime_count=sum(
            len(layout.placements) for layout in layouts if layout is not None
        ),
        dynamic_lifetime_count=sum(
            len(layout.dynamic_lifetimes) for layout in layouts if layout is not None
        ),
    )
