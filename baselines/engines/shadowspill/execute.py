"""Fresh worker: admit a saved Plan, initialize bounded storage, then measure."""

import os
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import torch.distributed as dist

from baselines.common import runner
from baselines.common.provenance import hardware_identity
from benchmarking.quickstart.phases.common import write
from shadowspill import Plan
from shadowspill.training import Trainer

from . import settings as engine_settings
from .initialize import Initialize
from .measurement import measure
from .reporting import comparison, describe, failure, memory, pool_record


def preflight(case, plan):
    from shadowspill.pytorch.distributed._bootstrap import local_resources, preflight
    from shadowspill.pytorch.distributed._resources import validate_resources

    spill = plan.memory_budgets.spill_bytes
    headroom = int(case.settings.host_headroom_gib * (1 << 30))
    if case.world > 1:
        assert dist.group.WORLD is not None
        preflight(dist.group.WORLD, case.device, spill, staging_bytes=headroom)
    else:
        validate_resources((local_resources(case.device, spill, headroom),))


def train(case, clock):
    prepared = case.resolved["phase_execution"]
    selected = prepared["selected"]
    saved_plan = Path(prepared["plan"])
    plan = Plan.load(saved_plan)
    geometry = replace(case.geometry, tokens_per_microbatch=int(selected["candidate"]))
    if case.world > 1:
        dist.init_process_group("gloo")
    case.stage = "admission"
    preflight(case, plan)
    with ExitStack() as stack:
        trainer = stack.enter_context(
            Trainer(
                plan,
                device=case.local_rank,
                external_headroom_bytes=int(
                    case.settings.external_headroom_gib * (1 << 30)
                ),
            )
        )
        case.record["hardware"] = hardware_identity()
        case.stage = "initialization"
        with Initialize(case.settings) as fill:
            trainer.initialize(fill)
            if fill.checkpoint is not None:
                case.record["checkpoint"] = {
                    "directory": str(Path(case.settings.checkpoint).resolve()),
                    "config": asdict(fill.checkpoint.config),
                }
                case.record["loaded"] = fill.report()
        runtime = trainer.executor.runtime
        case.record.update(
            plan={
                "file": str(saved_plan),
                "digest": selected["plan_digest"],
                "summary": plan.summary,
            },
            shared_preparation=prepared["artifact"],
            preparation_seconds=prepared["preparation_seconds"],
            microbatch_search={
                "candidates": list(case.settings.microbatch_candidates(case.world)),
                "orderings": case.settings.orderings,
                "chosen": {
                    "tokens_per_microbatch": geometry.tokens_per_microbatch,
                    "ordering": selected["ordering"],
                },
            },
            microbatches_per_step=geometry.accumulation,
            pools_after_initialization={
                name: pool_record(runtime.pool_statistics(name))
                for name in ("execution", "spill")
            },
        )
        case.save()
        write(case.directory / "plan.json", case.record["plan"])
        describe(case, plan, geometry, selected)
        config = case.settings.request().entry.config(**case.settings.model_config)
        data = case.data(config.vocab_size)
        case.stage = "training"
        observed = measure(case, clock, trainer, geometry, data, stack)
        pools = {
            name: pool_record(runtime.pool_statistics(name))
            for name in ("execution", "spill")
        }
    summary = plan.summary
    return runner.Outcome(
        evidence=observed["evidence"],
        reference_checks=observed["reference_checks"],
        peaks={
            "peak_allocated_bytes": pools["execution"]["peak_allocated_bytes"],
            "peak_reserved_bytes": plan.memory_budgets.execution_bytes,
        },
        result={
            "shadowspill_memory": {
                "execution_pool": pools["execution"],
                "spill_pool": pools["spill"],
                "predicted_device_peak_bytes": summary["execution_peak_bytes"],
                "predicted_spill_peak_bytes": summary["spill_peak_bytes"],
                "device_used_bytes_max": observed["device_used_bytes_max"],
            },
            "planning_seconds": prepared["preparation_seconds"],
            "simulation_comparison": comparison(plan, observed["seconds"], case.rank),
        },
        provenance={
            name: case.record[name]
            for name in (
                "shared_preparation",
                "plan",
                "microbatch_search",
                "pools_after_initialization",
            )
        },
        memory=memory,
        geometry=geometry.as_dict(),
    )


def main(argv=None):
    for name, value in {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
        os.environ.setdefault(name, value)
    runner.main(
        "shadowspill",
        engine_settings,
        train,
        argv,
        standalone=False,
        device_first=False,
        failure=failure,
    )


if __name__ == "__main__":
    main()
