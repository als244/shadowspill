"""Fresh Executor worker: load the exact saved Plan, initialize, warm up, capture."""

import time
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

from benchmarking.quickstart.phases.common import experiment, plan_identity, write
from benchmarking.quickstart.phases.distributed import workers
from shadowspill import Executor, Plan

from .selection import completed


@contextmanager
def capture_window(group, settle_seconds):
    # CPU barriers avoid queuing a collective while another rank drains capture.
    def barrier():
        if group.count > 1:
            dist.barrier()

    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("shadowspill.capture.initialize"):
        pass
    barrier()
    if group.rank == 0:
        torch.cuda.cudart().cudaProfilerStart()
    barrier()
    time.sleep(settle_seconds)
    barrier()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        barrier()
        time.sleep(settle_seconds)
        barrier()
        if group.rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
        barrier()


def run(args):
    request, index, selected = completed(
        args.quickstart, budget_gib=args.budget_gib, world_size=args.world_size
    )
    if len(selected) != 1:
        raise ValueError("a capture worker needs one selected budget")
    winner = selected[0]
    plan = Plan.load(args.quickstart / "plans" / winner["file"])
    if plan_identity(plan) != args.plan_digest:
        raise ValueError("saved Plan changed after capture selection")
    # Install the allocator before the model factory can touch CUDA.
    with (
        workers(request, index) as group,
        Executor(
            plan, device=int(request["device"].split(":")[-1]), rank_map=group.rank_map
        ) as executor,
        torch.cuda.device(executor.device),
    ):
        setup = experiment(
            request, logical_rank=None if group.count == 1 else group.logical_rank
        )
        with setup.get("context", nullcontext)():
            executor.initialize(setup["fill_tensor"])
        inputs = dict(setup["candidates"][winner["candidate"]])
        inputs["hyperparams"] = setup.get("hyperparams", {})
        for _ in range(args.warmup_steps):
            executor.run(**inputs).result()
        with capture_window(group, args.capture_settle_seconds):
            for step in range(
                args.warmup_steps + 1, args.warmup_steps + args.capture_steps + 1
            ):
                label = (
                    f"shadowspill.capture/budget-{args.budget_gib:g}GiB/"
                    f"rank-{group.logical_rank}/step-{step}"
                )
                with torch.cuda.nvtx.range(label):
                    executor.run(**inputs, profiler_annotations=True).result()
        write(
            args.output / f"rank-{group.logical_rank:05d}" / "exact-plan.json",
            {
                "status": "matched",
                "plan_digest": plan_identity(plan),
                "source": str(args.quickstart / "plans" / winner["file"]),
                "logical_rank": group.logical_rank,
                "warmup_steps": args.warmup_steps,
                "capture_steps": args.capture_steps,
                "performance_result": False,
            },
        )
