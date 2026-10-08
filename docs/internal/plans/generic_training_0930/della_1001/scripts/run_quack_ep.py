"""Bounded two-process QuackMoE training/capture probe; run with torchrun.

This experiment is outside the qualification suite. Each variant gets its own
processes and artifact store. Forced resolutions are a test-only search filter.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from mlops.optim import AdamW
from workloads.pytorch.olmoe import OLMoEConfig
from shadowspill.pytorch import ProfilingOptions
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill


def force_variant(variant):
    from shadowspill.ir import TaskAlternativeChoice
    from shadowspill.pytorch.distributed import _selection

    def choices(program, resolution_options):
        selected = []
        for group in program.task_alternative_groups:
            assert variant in {option.option_id for option in group.options}
            selected.append(TaskAlternativeChoice(group.group_id, variant))
        return (tuple(selected),)

    _selection.resolutions = choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--variant", choices=("save", "recompute"), default="save")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--router-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--share-buffer", action="store_true")
    parser.add_argument("--model-owned-buffer", action="store_true")
    args = parser.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    force_variant(args.variant)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 2
    assert args.tokens % 256 == 0
    directory = args.outdir / f"rank-{rank:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def log(event, **values):
        record = dict(
            utc=datetime.now(UTC).isoformat(), rank=rank, event=event,
            elapsed_seconds=time.monotonic() - started, **values,
        )
        print(json.dumps(record), flush=True)
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    config = OLMoEConfig(args.layers, 512, 4, 4, 128, 8, 2, 1024, 1024, max_seq_len=256)
    generator = torch.Generator().manual_seed(147 + rank)
    batches = [
        tuple(torch.randint(0, config.vocab_size, (args.tokens // 256, 256), generator=generator)
              for _ in range(2))
        for _ in range(args.steps)
    ]
    torch.save(batches, directory / "batches.pt")

    def objective(model, data):
        tokens, targets = data
        return model.loss(tokens, targets, reduction="sum", aux_coef=0.01) / (world * args.tokens)

    with ExitStack() as stack:
        dist.init_process_group("gloo", timeout=timedelta(seconds=900))
        stack.callback(dist.destroy_process_group)
        backend = stack.enter_context(ShadowSpill(
            device="auto", execution_gib=8, spill_gib=4,
            external_headroom_gib=1, control_group=dist.group.WORLD,
            artifact_store=directory / "artifacts", partition="whole",
            preparation_timeout=900,
            profiling_options=ProfilingOptions(
                conditioning_seconds=0, measurement_seconds=0, minimum_samples=3,
            ),
        ))
        # Quack's dependencies inspect the device while importing. Install the
        # allocator before loading them, just as before constructing resources.
        from mlops.expert_parallel import QuackMoEConfig as MoEConfig
        from mlops.expert_parallel import create_buffer
        from workloads.mlops import OLMoE

        group = dist.new_group(backend="nccl", device_id=backend.device,
                               timeout=timedelta(seconds=900))
        stack.callback(dist.destroy_process_group, group)
        options = MoEConfig(
            ep_size=world, num_experts=config.n_experts, top_k=config.top_k,
            model_dim=config.d_model, expert_hidden_dim=config.d_ff_expert,
            weight_grad_dtype=torch.bfloat16, renormalize_topk=False,
        )
        buffers = []
        for _ in range(0 if args.model_owned_buffer else (1 if args.share_buffer else config.n_layers)):
            buffer = create_buffer(options, args.tokens, group)
            stack.callback(buffer.destroy)
            buffers.append(buffer)
        if args.share_buffer:
            buffers *= config.n_layers
        torch.manual_seed(20261001 + rank)
        model = OLMoE(config, ep_group=group, **({"token_capacity": args.tokens} if args.model_owned_buffer else {"buffers": buffers}), device=backend.device, parameter_device="cpu", router_dtype=getattr(torch, args.router_dtype))
        stack.callback(model.close)
        expert_ids = {id(p) for p in model.expert_parameters()}
        expert_names = [name for name, p in model.named_parameters() if id(p) in expert_ids]
        parameter_names = [name for name, _ in model.named_parameters()]
        log("model_ready", config=asdict(config), device=str(backend.device),
            unique_expert_parameters=expert_names,
            parameters=sum(p.numel() for p in model.parameters()))
        trainer = stack.enter_context(Trainer(
            model, objective=objective, optimizer=AdamW,
            optimizer_args=dict(lr=0.0003, betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=0, opt_state_dtype=torch.float32,
                                gradient_dtype=torch.float32),
            grad_dtype=torch.float32, master_dtype=torch.float32,
            backend=backend,
            distributed=Distributed(group, replica_overrides=[(expert_names, None)],
                                    groups={"ep": group}, timeout=900),
        ))
        log("prepare_start", variant=args.variant)
        trainer.prepare(batches[0])
        report = trainer.plan
        selected = [choice.option_id for choice in report.search_result.selections]
        assert selected and set(selected) == {args.variant}
        (directory / "plan.json").write_text(json.dumps(report.execution_plan.to_dict(), indent=2) + "\n")
        metadata = dict(config=asdict(config), tokens=args.tokens, world=world,
                        parameters=parameter_names, experts=expert_names, selected=selected)
        (directory / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
        initial = trainer._execution.call.state_dict(weights="compute")
        torch.save(initial, directory / "initial.pt")
        log("prepared", selected=selected)
        trainer.diagnose(batches[0], directory=directory / "startup")
        log("startup_diagnostics_complete")
        losses = []
        for step, batch in enumerate(batches, 1):
            result = trainer.step(batch)
            assert math.isfinite(result.loss)
            state = trainer._execution.call.state_dict(weights="compute")
            peers = [None] * world
            dist.all_gather_object(peers, state["model"])
            for name in parameter_names:
                if name not in expert_names:
                    torch.testing.assert_close(peers[0][name], peers[1][name], rtol=0, atol=0)
            changed_experts = [name for name in expert_names
                               if not torch.equal(state["model"][name], initial["model"][name])]
            assert len(changed_experts) == len(expert_names)
            losses.append(result.loss)
            torch.save(state, directory / f"step-{step:03d}.pt")
            log("step_complete", step=step, local_loss=result.loss,
                replicated_parameters_identical=True, changed_experts=changed_experts)
        (directory / "summary.json").write_text(json.dumps(dict(
            status="passed_capture_and_training", variant=args.variant, losses=losses,
            numerical_oracle="pending separate comparison", **metadata,
        ), indent=2) + "\n")
        log("complete", variant=args.variant)


if __name__ == "__main__":
    main()
