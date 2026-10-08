"""One fresh-process full-model LoRA correctness or memory/throughput case.

Every case saves its config, parameter inventory, graphpairs, pool statistics,
process peak RSS and per-step timings. Use the sweep runner for resume.
"""
from __future__ import annotations

import argparse
import copy
import gc
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))

import torch
from mlops.dispatch import set_weight_gradient_dtype
from mlops.optim import AdamW
from shadowspill.ir import TaskAlternativeChoice
from shadowspill.task.profiling import ProfilingOptions
from shadowspill.training import Trainer
from shadowspill.training.backends import ShadowSpill
from workloads import mlops, pytorch
from workloads.lora import configure_lora, parameter_report

GIB = 1 << 30


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--family", choices=("llama3", "qwen35", "olmoe", "qwen3moe", "qwen35moe"), default="llama3")
    p.add_argument("--implementation", choices=("mlops", "pytorch"), default="mlops")
    p.add_argument("--mode", choices=("full", "lora", "lora_head"), default="lora")
    p.add_argument("--variant", choices=("save", "recompute"), default="save")
    p.add_argument("--preset", choices=("smoke", "benchmark", "1b"), default="smoke")
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--record-gc", action="store_true")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--warmup-seconds", type=float, default=1.0)
    p.add_argument("--execution-gib", type=float, default=4)
    p.add_argument("--spill-gib", type=float, default=None)
    p.add_argument("--outdir", type=Path)
    config, _ = p.parse_known_args()
    if config.config:
        p.set_defaults(**json.loads(config.config.read_text()))
    args = p.parse_args()
    if args.outdir is None:
        p.error("--outdir or an outdir in --config is required")
    args.outdir = Path(args.outdir)
    return args


def build(args):
    models = mlops if args.implementation == "mlops" else pytorch
    cls = {"llama3": "Llama3", "qwen35": "Qwen35", "olmoe": "OLMoE",
           "qwen3moe": "Qwen3MoE", "qwen35moe": "Qwen35MoE"}[args.family]
    if args.preset == "smoke":
        from tests.workloads.test_lora_models import tiny_model
        model = tiny_model(args.implementation, cls)
    else:
        width = 768
        if args.family == "llama3":
            config = models.Llama3Config(4, width, 12, 4, 2304, 32768, max_seq_len=args.sequence_length)
        elif args.family == "qwen35":
            config = models.Qwen35Config(4, width, 4, 12, 4, 64, 0.5, 4, 8, 64, 64, 4, 2304, 32768,
                                         max_seq_len=args.sequence_length)
        elif args.family == "olmoe":
            config = models.OLMoEConfig(4, width, 12, 4, 64, 32, 4, 512, 32768,
                                        max_seq_len=args.sequence_length)
        else:
            base = getattr(models, cls + "Config")()
            config = replace(base, n_layers=4, d_model=width, n_heads=12, n_kv_heads=4, head_dim=64,
                             n_experts=32, top_k=4, d_ff_expert=512, d_ff_shared=512 if base.d_ff_shared else 0,
                             vocab_size=32768, max_seq_len=args.sequence_length,
                             lin_k_heads=4, lin_v_heads=8, lin_k_head_dim=64, lin_v_head_dim=64)
        if args.preset == "1b":
            config = replace(getattr(models, cls + "Config").numerical(), max_seq_len=args.sequence_length)
        with torch.device("cpu"):
            model = getattr(models, cls)(config)
    if args.mode != "full":
        configure_lora(model, rank=args.rank, alpha=args.rank,
                       head="lora" if args.mode == "lora_head" else "frozen")
        if args.preset == "smoke":
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if "lora_" in name and name.endswith("b"):
                        value.normal_(std=0.02)
    return model


def objective(model, batch):
    tokens, targets, lengths = batch
    return model.loss(tokens, targets, seq_lens=lengths, reduction="sum") / targets.numel()


def pool_stats(backend):
    return {name: {field: int(getattr(s, field)) for field, _ in s._fields_}
            for name in ("execution", "spill")
            for s in (backend.runtime.pool_statistics(name),)}


def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def save_pairs(call, outdir):
    stages = call.plan_report.diagnostics.unique_stages
    data = [stage.as_dict() for stage in stages]
    (outdir / "graphpairs.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    table = ["| Stage | Variant | Direction | Inputs MiB | Mutated MiB | Outputs MiB | Workspace MiB | Total MiB | Runtime ms |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for stage in stages:
        for pair in stage.graph_pairs:
            for direction, profile in (("fwd", pair.forward), ("bwd", pair.backward)):
                if profile is None:
                    continue
                sizes = [profile.input_allocation_bytes, profile.mutation_allocation_bytes,
                         profile.output_allocation_bytes, profile.task_workspace_bytes]
                table.append(f"| {stage.unique_stage_id} | {pair.variant} | {direction} | " +
                             " | ".join(f"{size/(1<<20):.3f}" for size in [*sizes, sum(sizes)]) +
                             f" | {profile.runtime_ns/1e6:.4f} |")
    (outdir / "graphpairs.md").write_text("\n".join(table) + "\n")
    return data


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    gc_events = []
    def gc_event(phase, info):
        gc_events.append({"phase": phase, "time": time.perf_counter(), **info})
    if args.record_gc:
        gc.callbacks.append(gc_event)
    torch.set_num_threads(4)
    torch.manual_seed(207)
    torch.set_default_dtype(getattr(torch, args.dtype))
    set_weight_gradient_dtype(torch.float32)
    model = build(args)
    inventory = parameter_report(model)
    inventory["model_config"] = asdict(model.config)
    (args.outdir / "parameters.json").write_text(json.dumps(inventory, indent=2) + "\n")
    # Same rule in both modes: parameter bytes + FP32 gradient/moments for
    # trainable values, then 25% transient-state headroom and 2 GiB scratch.
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    persistent_estimate = parameter_bytes + 12 * inventory["trainable_parameters"]
    spill = args.spill_gib if args.spill_gib is not None else math.ceil(1.25*persistent_estimate/GIB + 2)
    options = {**vars(args), "outdir": str(args.outdir), "config": str(args.config) if args.config else None,
               "resolved_spill_gib": spill, "estimated_persistent_bytes": persistent_estimate}
    (args.outdir / "config.json").write_text(json.dumps(options, indent=2) + "\n")
    if args.tokens % args.sequence_length:
        raise ValueError("tokens must be divisible by sequence length")
    data_generator = torch.Generator().manual_seed(9763)
    data = (torch.randint(model.config.vocab_size, (1, args.tokens), generator=data_generator),
            torch.randint(model.config.vocab_size, (1, args.tokens), generator=data_generator),
            (args.sequence_length,) * (args.tokens // args.sequence_length))
    oracle = copy.deepcopy(model) if args.preset == "smoke" else None
    initial_trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad} if oracle else {}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad} if oracle else {}
    optimizer_type = torch.optim.SGD if oracle else AdamW
    optimizer_args = {"lr": 0.02, "foreach": False} if oracle else {
        "lr": 1e-4, "weight_decay": 0.0, "gradient_dtype": torch.float32,
        "opt_state_dtype": torch.float32, "parameter_rounding": "nearest", "opt_state_rounding": "nearest"}
    optimizer = optimizer_type((p for p in oracle.parameters() if p.requires_grad), **optimizer_args) if oracle else None
    def choices(program, _shares):
        return (tuple(TaskAlternativeChoice(group.group_id, args.variant)
                      for group in program.task_alternative_groups),)
    profiling = ProfilingOptions(warmup_iterations=2, conditioning_seconds=0 if oracle else 1.0,
                                  measurement_seconds=0 if oracle else 0.2)
    print("START", json.dumps(options), flush=True)
    print("PARAMETERS", inventory["total_parameters"], "TRAINABLE", inventory["trainable_parameters"], flush=True)
    begun = time.perf_counter()
    with (patch("shadowspill.planner.search.algorithms.pressurefit.resolutions", choices),
          ShadowSpill(device="cuda:0", execution_gib=args.execution_gib, spill_gib=spill,
                      artifact_store=args.outdir/"artifacts", profiling_options=profiling) as backend,
          Trainer(model, objective=objective, optimizer=optimizer_type, optimizer_args=optimizer_args,
                  grad_dtype=torch.float32, backend=backend,
                  hyperparams=() if oracle else ("lr",)) as trainer):
        trainer.prepare(data)
        preparation_seconds = time.perf_counter()-begun
        call = trainer._execution.call
        save_pairs(call, args.outdir)
        preparation_stats, preparation_rss = pool_stats(backend), rss()
        selections = [str(x) for x in call.plan_report.execution_plan.selections]
        (args.outdir/"selections.json").write_text(json.dumps(selections, indent=2))
        print("PREPARED", preparation_seconds, "seconds; peak RSS GiB", preparation_rss/GIB, flush=True)
        rows = []
        if not oracle:
            warm_started, i = time.perf_counter(), 0
            while i < args.warmup or time.perf_counter()-warm_started < args.warmup_seconds:
                # MLOps lr=0 executes the same optimizer kernels without
                # changing weights, moments, counters or stochastic RNG state.
                result = trainer.step(data, hyperparams={"lr": 0.0})
                i += 1
            print("WARMUP", i, "iterations", time.perf_counter()-warm_started, "seconds", flush=True)
        for step in range(args.steps):
            expected_loss = None
            if oracle:
                optimizer.zero_grad(set_to_none=True)
                expected = objective(oracle, data)
                expected.backward()
                optimizer.step()
                expected_loss = float(expected.detach())
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = trainer.step(data, hyperparams=None if oracle else {"lr": 1e-4})
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            if not math.isfinite(result.loss):
                raise AssertionError(f"nonfinite training loss at step {step}: {result.loss}")
            row = {"step": step, "started_at": started, "seconds": seconds, "reported_seconds": result.seconds,
                   "loss": result.loss, "reference_loss": expected_loss,
                   "tokens_per_second": args.tokens/seconds}
            rows.append(row)
            (args.outdir/"steps.json").write_text(json.dumps(rows, indent=2) + "\n")
            print("STEP", json.dumps(row), flush=True)
            if oracle:
                torch.testing.assert_close(torch.tensor(result.loss, dtype=torch.float32),
                                           torch.tensor(expected_loss, dtype=torch.float32),
                                           atol=3e-4 if args.dtype == "float32" else 1e-3,
                                           rtol=3e-4 if args.dtype == "float32" else 2e-3)
        measured_stats, measured_rss = pool_stats(backend), rss()
        numerical = None
        if oracle:
            state = call.state_dict()["model"]
            torch.testing.assert_close(state, oracle.state_dict(), atol=2e-4,
                                       rtol=3e-3 if args.dtype == "float32" else 1e-2)
            actual_updates = torch.cat([(state[n].float()-v.float()).reshape(-1) for n,v in initial_trainable.items()])
            reference_updates = torch.cat([(oracle.state_dict()[n].float()-v.float()).reshape(-1) for n,v in initial_trainable.items()])
            update_error = float((actual_updates-reference_updates).norm()/reference_updates.norm().clamp_min(1e-20))
            numerical = {"trainable_update_relative_l2_error": update_error,
                         "max_loss_relative_error": max(abs(r["loss"]-r["reference_loss"])/abs(r["reference_loss"]) for r in rows),
                         "max_loss_absolute_error": max(abs(r["loss"]-r["reference_loss"]) for r in rows)}
            assert update_error < (0.005 if args.dtype == "float32" else 0.05), numerical
            (args.outdir/"numerical.json").write_text(json.dumps(numerical, indent=2)+"\n")
            for name, value in frozen.items():
                torch.testing.assert_close(state[name], value, atol=0, rtol=0)
        median = statistics.median(row["seconds"] for row in rows)
        summary = {"status": "passed", "config": options,
                   "parameters": inventory["total_parameters"], "trainable": inventory["trainable_parameters"],
                   "parameter_bytes": parameter_bytes, "preparation_seconds": preparation_seconds,
                   "median_step_seconds": median, "tokens_per_second": args.tokens/median,
                   "peak_host_rss_preparation_bytes": preparation_rss,
                   "peak_host_rss_execution_bytes": measured_rss,
                   "pools_after_preparation": preparation_stats, "pools_after_execution": measured_stats,
                   "frozen_state_bitwise_checked": bool(oracle), "numerical": numerical, "steps": rows}
        (args.outdir/"result.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.record_gc:
            (args.outdir/"gc-events.json").write_text(json.dumps(gc_events, indent=2)+"\n")
            gc.callbacks.remove(gc_event)
        print("RESULT", json.dumps({key: summary[key] for key in
            ("status", "parameters", "trainable", "preparation_seconds", "median_step_seconds",
             "tokens_per_second", "peak_host_rss_execution_bytes")}), flush=True)


if __name__ == "__main__":
    main()
