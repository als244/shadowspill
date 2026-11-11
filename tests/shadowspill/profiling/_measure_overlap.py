"""Fresh build and distributed saved-code overlap measurement smoke."""

import json
import sys
from datetime import timedelta
from pathlib import Path

import torch

from shadowspill import (
    MemoryBudgets,
    OverlapProfileCache,
    Parallelism,
    ProfiledProgram,
    Profiles,
    apply_profiles,
    build_forward_backward,
    measure_communication,
    measure_overlap,
    measure_program,
    plan_program,
)


def main(mode, root):
    root.mkdir(parents=True, exist_ok=True)
    if mode == "build":
        model = torch.nn.Linear(1024, 1024, bias=False, device="meta")
        program = build_forward_backward(
            model,
            objective=lambda m, x: m(x).square().mean(),
            example_args=(torch.empty(2048, 1024),),
            num_examples=2,
            parallelism=Parallelism.dp(ranks_per_replica=2),
            partition="whole",
        )
        profiles = measure_program(program, warmup_seconds=0.005, measure_seconds=0.01)
        apply_profiles(program, profiles).save(root / "profiled.ssp")
        return

    import torch.distributed as dist

    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    try:
        profiled = ProfiledProgram.load(root / "profiled.ssp")
        info = measure_communication(
            program=profiled.program,
            device="cuda:0",
            rank_map={0: 1, 1: 0},
            sizes_bytes=(1024, 1 << 20),
            warmup_seconds=0.005,
            measure_seconds=0.01,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("overlap measurement tried to compile saved code")

        import torch._inductor.compile_fx as compiler
        import triton

        compiler.compile_fx = compiler.compile_fx_inner = triton.compile = forbidden
        cache = OverlapProfileCache(root / "cache", mode="refresh")
        overlap = measure_overlap(
            profiled, info, device="cuda:0", rank_map={0: 1, 1: 0}, cache=cache
        )
        assert overlap.metadata["overlap_calibration"]["attempted_tasks"] > 0
        observations = [
            row
            for profile in overlap.tasks.values()
            for row in profile.metadata["overlap_calibration"]["observations"]
        ]
        assert observations and all("per_rank" in row for row in observations)
        assert all(
            {r["logical_rank"] for r in row["per_rank"]} == {0, 1}
            for row in observations
        )
        assert all(peer == overlap.to_dict() for peer in _gather(overlap.to_dict()))
        # A fresh reader must reuse only globally agreed supported routes.
        # Unsupported routes may be retried; original timing/memory is untouched.
        repeated = measure_overlap(
            profiled,
            info,
            device="cuda:0",
            rank_map={0: 1, 1: 0},
            cache=OverlapProfileCache(root / "cache"),
        )
        assert all(peer == repeated.to_dict() for peer in _gather(repeated.to_dict()))
        assert len(repeated.metadata["overlap_calibration"]["cache"]["hits"]) == sum(
            row["status"] == "measured" for row in observations
        )
        for key, original in overlap.tasks.items():
            previous = {
                (cost.timing_class, cost.bytes): cost
                for cost in original.overlap_costs or ()
            }
            updated = {
                (cost.timing_class, cost.bytes): cost
                for cost in repeated.tasks[key].overlap_costs or ()
            }
            assert all(updated[item] == cost for item, cost in previous.items())
        merged = apply_profiles(profiled.program, profiled.profiles.merge(overlap))
        for key, original in profiled.task_profiles.items():
            updated = merged.task_profiles[key]
            assert updated.duration_seconds == original.duration_seconds
            assert updated.workspace_bytes == original.workspace_bytes
            assert updated.metadata.get("allocation_contract") == original.metadata.get(
                "allocation_contract"
            )
        plan = plan_program(
            merged,
            memory_budgets=MemoryBudgets(4 << 30, 1 << 30),
            communication_info=info,
        )
        classes = {
            phase.timing_class
            for alias in plan._costed_program.alias_groups
            if alias.transfer
            for phase in (*alias.transfer.fetch, *alias.transfer.evict)
        }
        costs = [
            cost
            for profile in overlap.tasks.values()
            for cost in profile.overlap_costs or ()
        ]
        assert all(cost.timing_class in classes for cost in costs)
        if dist.get_rank() == 0:
            overlap.save(root / "overlap.ssp")
            assert Profiles.load(root / "overlap.ssp").to_dict() == overlap.to_dict()
            merged.save(root / "profiled-overlap.ssp")
            info.save(root / "communication.ssp")
            plan.save(root / "plan.ssp")
            result = {
                "status": "PASS",
                "task_forms": len(overlap.tasks),
                "accepted_rates": len(costs),
                "observations": len(observations),
            }
            (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(result, flush=True)
    finally:
        dist.destroy_process_group()


def _gather(value):
    import torch.distributed as dist

    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
