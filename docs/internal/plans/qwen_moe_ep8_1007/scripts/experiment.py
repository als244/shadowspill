"""One Qwen EP geometry through the ordinary generic quickstart API.

Run with torchrun. The sweep script invokes this once per exact MoonEP token
capacity, combines search winners, then measures only the winning geometries.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from mlops.dispatch import set_weight_gradient_dtype
from mlops.optim import AdamW
from shadowspill.pytorch import Distributed, ProfilingOptions
from workloads.mlops import Qwen30B, Qwen30BConfig, Qwen35B, Qwen35BConfig


def integers(value):
    return [int(part) for part in value.split(",") if part]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen30b", "qwen35b"), required=True)
    parser.add_argument("--tokens-per-rank", type=int, required=True,
                        help="tokens in one microbatch on each rank")
    parser.add_argument("--global-tokens", type=int, default=1 << 22)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--search-budgets", type=integers, default=[20, 30, 40, 50, 60, 70])
    parser.add_argument("--run-budgets", type=integers, default=[],
                        help="empty for search only; search artifacts are retained")
    parser.add_argument("--spill-gib", type=float, default=64)
    parser.add_argument("--external-headroom-gib", type=float, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--symmetric-planning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--plan-store", type=Path)
    parser.add_argument("--seed", type=int, default=607)
    parser.add_argument("--tiny", action="store_true", help="small EP model for integration checks")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--profile-conditioning-seconds", type=float, default=1.0)
    parser.add_argument("--profile-measurement-seconds", type=float, default=0.3)
    return parser.parse_args()


def objective(global_tokens, model, tokens, targets, lengths):
    loss, metrics = model.loss(
        tokens, targets, seq_lens=lengths, reduction="sum",
        head_chunk_size=512, return_metrics=True,
    )
    return loss / global_tokens, metrics


def metrics(observed):
    # Called after the compiled step. There are no host reads in the objective.
    count = sum(item["trained_tokens"].item() for item in observed.metrics)
    ce = sum(item["ce_sum"].item() for item in observed.metrics) / count
    auxiliary = sum(item["auxiliary_sum"].item() for item in observed.metrics) / count
    return {"loss": ce, "auxiliary_loss": auxiliary, "work_units": count}


def factory(args):
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.global_tokens % (world * args.tokens_per_rank):
        raise ValueError("global tokens must divide into equal rank microbatches")
    if args.tokens_per_rank % args.sequence_length:
        raise ValueError("microbatch tokens must divide into whole sequences")
    cls, config_type = (Qwen30B, Qwen30BConfig) if args.model == "qwen30b" else (Qwen35B, Qwen35BConfig)
    config = replace(config_type(), max_seq_len=args.sequence_length)
    if args.tiny:
        config = replace(
            config, n_layers=4 if args.model == "qwen35b" else 2,
            d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
            n_experts=16, top_k=2, d_ff_expert=128,
            d_ff_shared=128 if config.d_ff_shared else 0, vocab_size=512,
            lin_k_heads=2, lin_v_heads=4, lin_k_head_dim=64, lin_v_head_dim=64,
        )
    accumulation = args.global_tokens // (world * args.tokens_per_rank)
    generator = torch.Generator().manual_seed(args.seed + rank)
    shape = (1, args.tokens_per_rank)
    examples = tuple(
        (torch.randint(config.vocab_size, shape, generator=generator),
         torch.randint(config.vocab_size, shape, generator=generator),
         (args.sequence_length,) * (args.tokens_per_rank // args.sequence_length))
        for _ in range(accumulation)
    )

    def setup(*, device):
        group = dist.new_group(backend="nccl", device_id=device, timeout=timedelta(seconds=3600))
        constructions = 0

        @contextmanager
        def resources():
            try:
                yield
            finally:
                dist.destroy_process_group(group)

        def model_factory():
            nonlocal constructions
            torch.manual_seed(args.seed + rank)
            model = cls(
                config, ep_group=group, token_capacity=args.tokens_per_rank,
                device=device, parameter_device="cpu", dtype=torch.bfloat16,
                weight_grad_dtype=torch.bfloat16,
            )
            # Experiment evidence only; model/planner code does not inspect
            # the backend's internal resource representation.
            from mlops.expert_parallel.quack.registry import _runtime

            runtimes = [_runtime(block.moe.experts._handle) for block in model.blocks]
            banks = {id(bank): bank for runtime in runtimes for bank in runtime.banks}
            buffers = {id(runtime.caller_buffer): runtime.caller_buffer for runtime in runtimes}
            assert len(banks) == 2 and len(buffers) == 1
            ctx = next(iter(buffers.values()))._require_ctx()
            summary = {
                "rank": rank, "parameters": sum(p.numel() for p in model.parameters()),
                "expert_parameters": sum(p.numel() for p in model.expert_parameters()),
                "moon_ep_buffers": len(buffers), "projection_banks": len(banks),
                "tokens_per_rank": int(ctx["S"]),
                "moon_ep_token_bytes_per_rank": sum(ctx[key].numel() * ctx[key].element_size() // world
                                                    for key in ("hidden_buf", "meta_buf")),
                "expert_bank_bytes_per_rank": sum(t.numel() * t.element_size() // world
                                                   for bank in banks.values() for t in bank.external_tensors()),
            }
            destination = args.outdir / f"rank-{rank:05d}" / f"model-{constructions:02d}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(summary, indent=2) + "\n")
            constructions += 1
            print("MODEL " + json.dumps(summary), flush=True)
            return model

        def distributed(model):
            return Distributed(
                group, replica_overrides=[(model.expert_parameters(), None)],
                groups={"ep": group}, timeout=3600,
                symmetric_planning=args.symmetric_planning,
            )

        return {
            "model_factory": model_factory,
            "cleanup_model": lambda model: model.close(),
            "objective": partial(objective, args.global_tokens),
            "optimizer": partial(AdamW, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                                 weight_decay=0.0, opt_state_dtype=torch.bfloat16),
            "hyperparams": {"lr": 3e-4},
            "plan_options": {"grad_dtype": torch.bfloat16},
            "metric_reducer": metrics,
            "candidates": {str(args.tokens_per_rank): examples},
            "distributed": distributed,
            "context": resources,
            "units_per_step": args.global_tokens // world,
            "unit_label": "tokens",
            "metadata": {
                "model": args.model, "model_config": asdict(config),
                "global_tokens_per_step": args.global_tokens,
                "tokens_per_microbatch_per_rank": args.tokens_per_rank,
                "microbatches_per_rank": accumulation,
                "world_size": world, "rank": rank, "units_scope": "per_rank",
                "precision": "bf16", "activation_transport": "bf16",
                "input_data": "deterministic synthetic tokens for quickstart",
            },
        }

    setup.__name__ = f"mlops_{args.model}_ep{world}_t{args.tokens_per_rank}"
    return setup


def main():
    args = parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    set_weight_gradient_dtype(torch.bfloat16)
    dist.init_process_group("gloo", timeout=timedelta(seconds=3600))
    try:
        from benchmarking.quickstart import run

        result = run(
            factory(args), search_budget_gib=args.search_budgets,
            run_budget_gib=args.run_budgets, spill_gib=args.spill_gib,
            steps=args.steps, output_dir=args.outdir, artifact_store=args.artifact_store,
            plan_store=args.plan_store,
            control_group=dist.group.WORLD, preparation_timeout=3600,
            search_workers=args.search_workers,
            external_headroom_gib=args.external_headroom_gib,
            plots=not args.no_plots, resolution_plans=True,
            profiling_options=ProfilingOptions(
                warmup_iterations=3,
                conditioning_seconds=args.profile_conditioning_seconds,
                measurement_seconds=args.profile_measurement_seconds,
            ),
        )
        if result:
            raise SystemExit(result)
        target = args.outdir / f"rank-{dist.get_rank():05d}" / "completed.json"
        target.write_text(json.dumps({"status": "passed", "rank": dist.get_rank()}) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
