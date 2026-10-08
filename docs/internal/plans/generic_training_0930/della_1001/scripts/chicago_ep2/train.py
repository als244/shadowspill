"""Chicago's sparse OLMoE recipe, with two-rank QuackMoE expert parallelism.

The JSON config controls the short run length separately from the original
9,537-update LR schedule. Model/data/optimizer construction stays caller-side.
"""

# ruff: noqa: E402 -- direct experiment entrypoint imports repository workloads.
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import sys
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from mlops.dispatch import set_deterministic_kernels, set_weight_gradient_dtype
from mlops.optim import AdamW

from shadowspill.planner import SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import (
    PressureFit,
    PressureFitOptions,
)
from shadowspill.pytorch import ProfilingOptions
from shadowspill.search.geometries import default_orderings
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.logging import DistributedLogger
from shadowspill.training.observations import MetricSummary, parameter_norms
from shadowspill.training.schedules import WarmupCosine
from workloads.pytorch.olmoe import OLMoEConfig
from workloads.recipes.text.data import PackedTokens
from workloads.recipes.text.olmoe_metrics import reduce_metrics
from workloads.recipes.text.source import PackedUpdates, validation_update


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class EPUpdates:
    """Partition a deterministic global packed update, normalizing globally."""

    def __init__(self, data, config, rank, world):
        self.rank, self.world = rank, world
        self.name = "packed"
        self.global_source = PackedUpdates(
            data,
            name=self.name,
            tokens=config["microbatch_tokens"],
            accumulation=config["tokens_per_step"] // config["microbatch_tokens"],
            max_seq_len=config["max_seq_len"],
        )

    def partition(self, update):
        parts = update["candidates"][self.name]
        assert len(parts) % self.world == 0
        return {
            "parts": parts[self.rank :: self.world],
            "normalizer": update["normalizer"],
        }

    def __iter__(self):
        return self

    def __next__(self):
        return self.partition(next(self.global_source))

    def state_dict(self):
        return self.global_source.state_dict()

    def load_state_dict(self, state):
        self.global_source.load_state_dict(state)


def microbatches(update):
    for values in update["parts"]:
        yield values, 1.0 / update["normalizer"]


def metrics(observed):
    summary = reduce_metrics(observed.metrics)
    count = sum(v["trained_tokens"].item() for v in observed.metrics)
    return MetricSummary({**summary.scalars, "work_units": count}, summary.tables)


def aggregate(records):
    result = {"step": records[0]["step"]}
    for name in ("train/loss", "train/work_units", "eval/loss"):
        if all(name in row for row in records):
            result[name] = sum(row[name] for row in records)
    for name in ("train/step_seconds", "train/elapsed_seconds", "eval/seconds"):
        if all(name in row for row in records):
            result[name] = max(row[name] for row in records)
    if "train/work_units" in result:
        total = result["train/work_units"]
        result["train/units_per_second"] = total / result["train/step_seconds"]
        for key in ("cross_entropy", "auxiliary", "weighted_auxiliary", "total"):
            name = "train/loss/" + key
            result[name] = (
                sum(row[name] * row["train/work_units"] for row in records) / total
            )
    for name, value in records[0].items():
        if name.startswith("hyperparameters/"):
            result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--profile-steps", type=int)
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-checkpoint", type=Path)
    args = parser.parse_args()
    if args.profile_steps is not None and (
        args.profile_steps < 1 or args.profile_warmup < 1 or args.plan_only
    ):
        parser.error("Profiling requires positive steps/warmup and no --plan-only")
    cfg = json.loads(args.config.read_text())
    # Reject recipe/API drift before allocating pools or constructing the model.
    inspect.signature(AdamW).bind([], **cfg["optimizer_args"])
    if args.steps is not None:
        cfg["steps"] = args.steps
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == cfg["world_size"]
    assert cfg["tokens_per_step"] % (world * cfg["microbatch_tokens"]) == 0
    root = Path(cfg["outdir"])
    rank_dir = root / f"rank-{rank:05d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    if (rank_dir / "metrics.jsonl").exists():
        raise FileExistsError("Refusing to append fresh training to an earlier run")

    def phase(name, **extra):
        event = dict(utc=datetime.now(UTC).isoformat(), rank=rank, phase=name, **extra)
        print(json.dumps(event), flush=True)
        with (rank_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def orderings(accumulation):
        maximum = cfg.get("planning_max_breadth")
        return tuple(
            ordering
            for ordering in default_orderings(accumulation)
            if maximum is None or ordering.breadth <= maximum
        )

    data = PackedTokens(cfg["data"], long_documents="splice", window=1024)
    source = EPUpdates(data, cfg, rank, world)
    initial_source = copy.deepcopy(source.state_dict())
    example = next(source)
    source.load_state_dict(initial_source)
    model_config = OLMoEConfig(**cfg["model"])
    set_deterministic_kernels(True)
    set_weight_gradient_dtype(torch.bfloat16)
    write_json(rank_dir / "config.json", cfg)
    with ExitStack() as stack:
        dist.init_process_group("gloo", timeout=timedelta(seconds=1800))
        stack.callback(dist.destroy_process_group)
        phase("runtime")
        backend = stack.enter_context(
            ShadowSpill(
                device="auto",
                execution_gib=cfg["execution_gib"],
                spill_gib=cfg["spill_gib"],
                external_headroom_gib=cfg["external_headroom_gib"],
                control_group=dist.group.WORLD,
                artifact_store=Path(cfg.get("artifact_store", root))
                / f"rank-{rank:05d}"
                / "artifacts",
                preparation_timeout=1800,
                profiling_options=ProfilingOptions(**cfg["profiling"]),
                search_options=SearchOptions(
                    algorithm=PressureFit(
                        PressureFitOptions(
                            resolution_options=tuple(
                                Fraction(v) for v in cfg["resolution_options"]
                            )
                        )
                    ),
                    workers=cfg["planning_workers"],
                ),
                orderings=orderings,
            )
        )
        from moonep.buffer import get_vmm_granularity

        from workloads.mlops import OLMoE

        group = dist.new_group(
            backend="nccl", device_id=backend.device, timeout=timedelta(seconds=1800)
        )
        stack.callback(dist.destroy_process_group, group)
        phase("model_initialization", vmm_granularity=int(get_vmm_granularity()))
        torch.manual_seed(cfg["seed"] + rank)
        model = OLMoE(
            model_config,
            ep_group=group,
            token_capacity=cfg["microbatch_tokens"],
            device=backend.device,
            parameter_device="cpu",
            router_dtype=torch.bfloat16,
        )
        stack.callback(model.close)
        # Match the Chicago workload's initialization distributions. EP shards
        # draw independent values; this is a fresh run, not a checkpoint replay.
        with torch.no_grad():
            for block in model.blocks:
                expert = block.moe.experts
                torch.nn.init.kaiming_uniform_(expert.router_weight, a=math.sqrt(5))
                expert.gate_up_weight.normal_(std=model_config.d_model**-0.5)
                expert.down_weight.normal_(std=model_config.d_ff_expert**-0.5)
        # Experiment evidence: verify that depth does not multiply scratch.
        from mlops.expert_parallel.quack.registry import _runtime

        runtimes = [_runtime(block.moe.experts._handle) for block in model.blocks]
        banks = {id(bank): bank for runtime in runtimes for bank in runtime.banks}
        buffers = {
            id(runtime.caller_buffer): runtime.caller_buffer for runtime in runtimes
        }
        assert len(banks) == 2 and len(buffers) == 1
        context = next(iter(buffers.values()))._require_ctx()
        communication = dict(
            token_capacity=context["S"],
            token_buffers=len(buffers),
            projection_banks=len(banks),
            expert_bank_bytes_per_rank=sum(
                tensor.numel() * tensor.element_size() // world
                for bank in banks.values()
                for tensor in bank.external_tensors()
            ),
            token_buffer_external_bytes_per_rank=sum(
                context[name].numel() * context[name].element_size() // world
                for name in ("hidden_buf", "meta_buf")
            ),
        )
        del banks, buffers, runtimes, context
        write_json(rank_dir / "communication-memory.json", communication)
        phase(
            "model_ready",
            local_parameters=sum(p.numel() for p in model.parameters()),
            gpu_free_bytes=torch.cuda.mem_get_info(backend.device)[0],
            communication=communication,
        )

        def objective(model, values):
            tokens, targets, lengths = values
            return model.loss(
                tokens,
                targets,
                seq_lens=lengths,
                reduction="sum",
                aux_coef=cfg["aux_coef"],
                return_metrics=True,
            )

        optimizer_args = dict(cfg["optimizer_args"])
        for key in ("gradient_dtype", "opt_state_dtype"):
            optimizer_args[key] = getattr(torch, optimizer_args[key])
        trainer = stack.enter_context(
            Trainer(
                model,
                objective=objective,
                microbatches=microbatches,
                backend=backend,
                optimizer=AdamW,
                optimizer_args=optimizer_args,
                schedules={
                    "lr": WarmupCosine(
                        **cfg["schedule"], total_steps=cfg["schedule_total_steps"]
                    )
                },
                master_dtype=None,
                grad_dtype=torch.bfloat16,
                parameter_metrics=parameter_norms,
                metric_reducer=metrics,
                distributed=Distributed(
                    group,
                    replica_overrides=[(model.expert_parameters(), None)],
                    groups={"ep": group},
                    timeout=1800,
                ),
            )
        )
        phase("planning")
        trainer.prepare(example)
        predicted_seconds = trainer.plan.search_result.simulation.makespan_ns / 1e9
        plan_record = dict(
            passed=True,
            tokens_per_rank=cfg["microbatch_tokens"],
            accumulation_per_rank=len(example["parts"]),
            predicted_seconds=predicted_seconds,
            execution_gib=cfg["execution_gib"],
            external_headroom_gib=cfg["external_headroom_gib"],
        )
        write_json(rank_dir / "planning.json", plan_record)
        write_json(
            rank_dir / "plan-diagnostics.json", trainer.plan.diagnostics.as_dict()
        )
        (rank_dir / "plan.json").write_text(
            trainer.plan.execution_plan.to_json() + "\n"
        )
        if trainer.planning is not None:
            trainer.planning.save(rank_dir / "search.json")
        phase("planned", **plan_record)
        if args.plan_only:
            return
        if args.profile_steps is not None:
            from nsys_profile import profile_trainer

            profile_trainer(
                trainer,
                source,
                rank_dir=rank_dir,
                steps=args.profile_steps,
                warmup=args.profile_warmup,
                checkpoint=args.profile_checkpoint,
                phase=phase,
            )
            return

        def evaluation():
            update = validation_update(
                data,
                name=source.name,
                tokens=cfg["microbatch_tokens"],
                max_seq_len=cfg["max_seq_len"],
                microbatches=cfg["eval_batches"],
            )
            return [source.partition(update)]

        # Admit and exercise evaluation before spending time on real updates.
        # Its compiled forward and physical layout differ from the training plan.
        phase("evaluation_preflight")
        evaluated = trainer.evaluate(evaluation)
        write_json(
            rank_dir / "evaluation-preflight.json",
            dict(loss=evaluated.mean_loss, seconds=evaluated.seconds),
        )
        phase("evaluation_ready", loss=evaluated.mean_loss)

        startup_diagnostics = True
        if cfg.get("training_end_utc"):
            startup = trainer.diagnose(example, directory=rank_dir / "startup")
            startup_diagnostics = False
            seconds = torch.tensor(
                [max(predicted_seconds, startup["traced_seconds"])],
                dtype=torch.float64,
                device="cpu",
            )
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            decision = [None]
            if rank == 0:
                remaining = (
                    datetime.fromisoformat(cfg["training_end_utc"]) - datetime.now(UTC)
                ).total_seconds()
                # The deadline already reserves checkpoint/sync time. Leave an
                # additional 15% for data preparation, evaluation and timing drift.
                count = min(
                    cfg["steps"], math.floor(remaining / (1.15 * seconds.item()))
                )
                if count >= 100:
                    count = count // 100 * 100
                decision[0] = count
            dist.broadcast_object_list(decision, src=0)
            if decision[0] < 1:
                raise RuntimeError(
                    "The allocation has no training time left after preparation"
                )
            cfg["steps"] = decision[0]
            window = dict(
                steps=cfg["steps"],
                seconds_per_step_estimate=seconds.item(),
                training_end_utc=cfg["training_end_utc"],
                schedule_total_steps=cfg["schedule_total_steps"],
            )
            write_json(rank_dir / "training-window.json", window)
            phase("training_window", **window)

        logger = stack.enter_context(
            DistributedLogger(
                dist.group.WORLD,
                run_dir=root,
                device=backend.device,
                reduce=aggregate,
                wandb=dict(
                    project=cfg["wandb_project"],
                    mode="online",
                    group=cfg.get("wandb_group", root.name),
                    config=cfg,
                ),
            )
        )
        write_json(
            rank_dir / "wandb.json",
            dict(
                rank_url=logger.local.run.url,
                aggregate_url=logger.aggregate.run.url if logger.aggregate else None,
            ),
        )

        def check_step(_trainer, result):
            if not math.isfinite(result.loss):
                raise FloatingPointError(f"Nonfinite loss at update {result.step}")
            phase(
                "step_completed",
                step=result.step,
                local_loss=result.loss,
                seconds=result.seconds,
                lr=result.hyperparams["lr"],
            )

        trainer.fit(
            source,
            steps=cfg["steps"],
            run_dir=root,
            logger=logger,
            startup_diagnostics=startup_diagnostics,
            callbacks=[check_step],
            tables_every=100,
            eval_data=evaluation,
            eval_every=cfg["eval_every"],
            checkpoint_every=cfg["checkpoint_every"],
            checkpoint_dir=root / "checkpoints",
            checkpoint_weights="compute",
        )
        write_json(
            rank_dir / "completed.json",
            dict(passed=True, steps=trainer.step_count, requested_steps=cfg["steps"]),
        )
        phase("complete", steps=trainer.step_count)


if __name__ == "__main__":
    main()
