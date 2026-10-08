"""Short real-data EP training validation after the complete quickstart sweep."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from mlops.dispatch import set_weight_gradient_dtype
from mlops.optim import AdamW
from shadowspill.planner import SearchOptions, StepDataOrdering
from shadowspill.pytorch import ProfilingOptions
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.logging import DistributedLogger
from workloads.mlops import Qwen30B, Qwen30BConfig, Qwen35B, Qwen35BConfig


class Updates:
    """Fixed 1K causal sequences, disjoint rank slices of the same global update."""

    def __init__(self, path, *, global_tokens, microbatch, rank, world):
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.global_tokens, self.microbatch = global_tokens, microbatch
        self.rank, self.world, self.step = rank, world, 0
        self.local = global_tokens // world
        if global_tokens % (world * microbatch) or microbatch % 1024:
            raise ValueError("batch must divide into rank microbatches and 1K sequences")

    def __iter__(self):
        return self

    def __next__(self):
        start = self.step * self.global_tokens + self.rank * self.local
        if start + self.local + 1 > len(self.tokens):
            raise StopIteration
        data = torch.from_numpy(np.asarray(self.tokens[start:start + self.local + 1], dtype=np.int64))
        self.step += 1
        return [
            (data[i:i + self.microbatch].reshape(1, -1).clone(),
             data[i + 1:i + self.microbatch + 1].reshape(1, -1).clone(),
             (1024,) * (self.microbatch // 1024))
            for i in range(0, self.local, self.microbatch)
        ]

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = int(state["step"])


def objective(model, values):
    tokens, targets, lengths = values
    return model.loss(tokens, targets, seq_lens=lengths, reduction="sum",
                      head_chunk_size=512, return_metrics=True)


def metrics(observed):
    count = sum(item["trained_tokens"].item() for item in observed.metrics)
    return {
        "cross_entropy": sum(item["ce_sum"].item() for item in observed.metrics) / count,
        "auxiliary_loss": sum(item["auxiliary_sum"].item() for item in observed.metrics) / count,
        "work_units": count,
    }


def aggregate(records):
    seconds = max(item["train/step_seconds"] for item in records)
    count = sum(item["train/work_units"] for item in records)
    result = {
        "step": records[0]["step"], "train/step_seconds": seconds,
        "train/work_units": count, "train/units_per_second": count / seconds,
        "train/loss": sum(item["train/loss"] for item in records),
    }
    for name in ("cross_entropy", "auxiliary_loss"):
        key = "train/" + name
        result[key] = sum(item[key] * item["train/work_units"] for item in records) / count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen30b", "qwen35b"), required=True)
    parser.add_argument("--tokens-per-rank", type=int, required=True)
    parser.add_argument("--execution-gib", type=float, required=True)
    parser.add_argument("--spill-gib", type=float, default=64)
    parser.add_argument("--global-tokens", type=int, default=1 << 22)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--symmetric-planning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--breadth", type=int, required=True, help="winner's breadth from quickstart")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    set_weight_gradient_dtype(torch.bfloat16)
    dist.init_process_group("gloo", timeout=timedelta(seconds=3600))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = args.outdir / f"rank-{rank:05d}"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "metrics.jsonl").exists():
        raise FileExistsError(f"fresh training cannot append to {root}")
    source = Updates(args.data / "train.bin", global_tokens=args.global_tokens,
                     microbatch=args.tokens_per_rank, rank=rank, world=world)
    example = next(source)
    source.load_state_dict({"step": 0})
    cls, config_type = (Qwen30B, Qwen30BConfig) if args.model == "qwen30b" else (Qwen35B, Qwen35BConfig)
    with ExitStack() as stack:
        stack.callback(dist.destroy_process_group)
        backend = stack.enter_context(ShadowSpill(
            execution_gib=args.execution_gib, spill_gib=args.spill_gib,
            external_headroom_gib=4,
            artifact_store=args.artifact_store / f"rank-{rank:05d}",
            control_group=dist.group.WORLD, preparation_timeout=3600,
            search_options=SearchOptions(workers=4), profiling_options=ProfilingOptions(),
            orderings=lambda n: (StepDataOrdering(n // args.breadth, args.breadth),),
        ))
        group = dist.new_group(backend="nccl", device_id=backend.device, timeout=timedelta(seconds=3600))
        stack.callback(dist.destroy_process_group, group)
        torch.manual_seed(607 + rank)
        model = cls(replace(config_type(), max_seq_len=1024), ep_group=group,
                    token_capacity=args.tokens_per_rank, device=backend.device,
                    parameter_device="cpu", dtype=torch.bfloat16,
                    weight_grad_dtype=torch.bfloat16)
        stack.callback(model.close)
        trainer = stack.enter_context(Trainer(
            model, objective=objective, backend=backend,
            microbatches=lambda update: ((part, 1.0 / args.global_tokens) for part in update),
            optimizer=AdamW, optimizer_args={
                "lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0,
                "gradient_dtype": torch.bfloat16, "opt_state_dtype": torch.bfloat16,
                "parameter_rounding": "stochastic", "opt_state_rounding": "stochastic",
            },
            schedules={"lr": lambda step: 3e-4}, grad_dtype=torch.bfloat16,
            metric_reducer=metrics,
            distributed=Distributed(group, replica_overrides=[(model.expert_parameters(), None)],
                                    groups={"ep": group}, timeout=3600,
                                    symmetric_planning=args.symmetric_planning),
        ))
        trainer.prepare(example)
        predicted = trainer.plan.search_result.simulation.makespan_ns / 1e9
        (root / "planning.json").write_text(json.dumps({"predicted_seconds": predicted}) + "\n")
        logger = stack.enter_context(DistributedLogger(
            dist.group.WORLD, run_dir=args.outdir, device=backend.device, reduce=aggregate,
        ))

        def check_step(_trainer, result):
            if not math.isfinite(result.loss):
                raise FloatingPointError(f"non-finite loss at step {result.step}")
            print(json.dumps({"rank": rank, "step": result.step, "loss": result.loss,
                              "seconds": result.seconds, "predicted_seconds": predicted}), flush=True)

        trainer.fit(source, steps=args.steps, run_dir=args.outdir,
                    startup_diagnostics=True, logger=logger, callbacks=(check_step,))
        (root / "completed.json").write_text(json.dumps({"status": "passed", "steps": args.steps}) + "\n")


if __name__ == "__main__":
    main()
