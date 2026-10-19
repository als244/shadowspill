"""torchrun: public sharded training on dense and MoE MLOps models."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from shadowspill.pytorch import ProfilingOptions
from shadowspill.training import Trainer
from shadowspill.training.backends import ShadowSpill
from workloads.recipes.text.distributed import data_parallel_groups


def model_for(name):
    torch.manual_seed(123)
    if name == "mlops_llama3":
        from workloads.mlops import Llama3
        from workloads.pytorch import Llama3Config

        model = Llama3(Llama3Config(2, 128, 4, 2, 256, 128, max_seq_len=32))
    elif name == "mlops_olmoe":
        from workloads.mlops import OLMoE
        from workloads.pytorch import OLMoEConfig

        model = OLMoE(OLMoEConfig(2, 128, 4, 2, 128, 8, 2, 128, 128, max_seq_len=32))
    else:
        from workloads.lora import configure_lora
        from workloads.mlops.qwen3_moe import Qwen3MoE, Qwen3MoEConfig

        model = Qwen3MoE(
            Qwen3MoEConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                n_kv_heads=2,
                head_dim=32,
                n_experts=8,
                top_k=2,
                d_ff_expert=128,
                vocab_size=128,
                max_seq_len=32,
            )
        )
        if name == "mlops_qwen3moe_lora":
            model = configure_lora(
                model, rank=16, alpha=16, factor_dtype=torch.bfloat16
            )
    return model.to(dtype=torch.bfloat16)


def objective(model, data):
    tokens, targets, denominator = data
    return model.loss(tokens, targets, reduction="sum") / denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "mlops_llama3",
            "mlops_olmoe",
            "mlops_qwen3moe",
            "mlops_qwen3moe_lora",
        ),
        required=True,
    )
    parser.add_argument("--shard-size", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--opt-state-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from mlops.dispatch import set_weight_gradient_dtype

    set_weight_gradient_dtype(torch.float32)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    args.out.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("gloo", timeout=timedelta(seconds=600))
    try:
        with ShadowSpill(
            device="auto",
            execution_gib=4,
            spill_gib=2,
            control_group=dist.group.WORLD,
            preparation_timeout=600,
            artifact_store=args.out / f"rank-{rank:05d}" / "artifacts",
            profiling_options=ProfilingOptions(
                conditioning_seconds=0,
                measurement_seconds=0,
                minimum_samples=3,
            ),
        ) as backend:
            distributed, resources = data_parallel_groups(
                device=backend.device,
                shard_size=args.shard_size,
                shard_gradients=True,
                timeout=600,
            )
            with resources():
                source = model_for(args.model)
                reference = copy.deepcopy(source).to(backend.device)
                reference_parameters = dict(reference.named_parameters())
                masters = {
                    name: torch.nn.Parameter(
                        value.detach().float().clone(),
                        requires_grad=value.requires_grad,
                    )
                    for name, value in reference_parameters.items()
                }
                options = dict(
                    lr=0.001,
                    betas=(0.8, 0.95),
                    eps=1e-4,
                    weight_decay=0.01,
                    foreach=False,
                )
                oracle = torch.optim.AdamW(masters.values(), **options)
                optimizer_type = torch.optim.AdamW
                optimizer_options = options
                if args.trace:
                    from mlops.optim import AdamW

                    optimizer_type = AdamW
                    optimizer_options = {
                        **options,
                        "gradient_dtype": torch.float32,
                        "opt_state_dtype": getattr(torch, args.opt_state_dtype),
                    }

                def data(step):
                    generator = torch.Generator().manual_seed(1000 + 101 * rank + step)
                    return (
                        torch.randint(0, 128, (2, 32), generator=generator),
                        torch.randint(0, 128, (2, 32), generator=generator),
                        float(64 * world),
                    )

                with Trainer(
                    source,
                    objective=objective,
                    optimizer=optimizer_type,
                    optimizer_args=optimizer_options,
                    backend=backend,
                    distributed=distributed,
                    master_dtype=torch.float32,
                    grad_dtype=torch.float32,
                    shard_optimizer=True,
                ) as trainer:
                    trainer.prepare(data(0))
                    if args.trace:
                        trainer.diagnose(
                            data(0), directory=args.out / f"trace-rank-{rank}", warmup=0
                        )
                        trace = json.loads(
                            (args.out / f"trace-rank-{rank}" / "step.json").read_text()
                        )["collectives"]
                        assert trace["complete"] and trace["records"], trace
                        assert {item["source"] for item in trace["records"]} >= {
                            "task",
                            "fetch",
                        }
                        assert all(
                            item["execution_task_id"] for item in trace["records"]
                        )
                        assert all(
                            item["alias_group_id"]
                            for item in trace["records"]
                            if item["source"] == "fetch"
                        )
                    maximum_error = 0.0
                    for step in range(args.steps):
                        batch = data(step)
                        reference.zero_grad(set_to_none=True)
                        oracle.zero_grad(set_to_none=True)
                        loss = objective(
                            reference,
                            (
                                batch[0].to(backend.device),
                                batch[1].to(backend.device),
                                batch[2],
                            ),
                        )
                        loss.backward()
                        for name, parameter in reference_parameters.items():
                            if not parameter.requires_grad:
                                continue
                            assert parameter.grad is not None, name
                            gradient = parameter.grad.float()
                            dist.all_reduce(gradient, group=distributed.group)
                            masters[name].grad = gradient
                        oracle.step()
                        with torch.no_grad():
                            for name, parameter in reference_parameters.items():
                                if parameter.requires_grad:
                                    parameter.copy_(masters[name])
                        result = trainer.step(batch)
                        actual = trainer._execution.call.state_dict(weights="compute")
                        torch.testing.assert_close(
                            torch.tensor(result.loss),
                            loss.detach().cpu(),
                            rtol=0.02,
                            atol=0.002,
                        )
                        for name, parameter in reference_parameters.items():
                            expected = parameter.detach().cpu()
                            error = (
                                (actual["model"][name].float() - expected.float())
                                .abs()
                                .max()
                                .item()
                            )
                            maximum_error = max(maximum_error, error)
                            torch.testing.assert_close(
                                actual["model"][name], expected, rtol=0.03, atol=0.002
                            )
                        print(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "model": args.model,
                                    "step": step + 1,
                                    "maximum_error": maximum_error,
                                    "passed": True,
                                }
                            ),
                            flush=True,
                        )
                    record = {
                        "rank": rank,
                        "model": args.model,
                        "steps": args.steps,
                        "transfer_shard_size": args.shard_size,
                        "optimizer_shard_size": world,
                        "maximum_parameter_error": maximum_error,
                        "passed": True,
                    }
                    (args.out / f"result-rank-{rank}.json").write_text(
                        json.dumps(record, indent=2) + "\n"
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
