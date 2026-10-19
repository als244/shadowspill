"""Write an explicit launch command for the pinned eight-H100 Qwen recipe."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from baselines.common import accumulation_steps, write_json

BRIDGE_REVISION = "6dea15d7fd499975ebf91fbeaf8c2b740a73d679"
CORE_REVISION = "ce852bfb93f364f64a556602fc71a5d9ed2c377e"
RECIPE = "qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--microbatch-tokens", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--global-tokens", type=int, default=2097152)
    parser.add_argument("--tensor-parallel", type=int, default=4)
    parser.add_argument("--pipeline-parallel", type=int, default=2)
    parser.add_argument("--expert-parallel", type=int, default=4)
    parser.add_argument("--expert-tensor-parallel", type=int, default=1)
    parser.add_argument(
        "--recompute", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args(argv)
    if (
        min(
            args.world_size,
            args.tensor_parallel,
            args.pipeline_parallel,
            args.expert_parallel,
            args.expert_tensor_parallel,
            args.warmup_steps,
            args.steps,
        )
        < 1
    ):
        parser.error("parallelism and step counts must be positive")
    dp, remainder = divmod(
        args.world_size, args.tensor_parallel * args.pipeline_parallel
    )
    expert_group = (
        args.expert_parallel * args.expert_tensor_parallel * args.pipeline_parallel
    )
    if remainder or args.world_size % expert_group:
        parser.error("world size must divide into complete attention and expert groups")
    accumulation = accumulation_steps(
        args.global_tokens, args.sequence_length, args.microbatch_tokens, dp
    )
    root, output = args.bridge_root.resolve(), args.output.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != BRIDGE_REVISION:
        parser.error("Megatron Bridge checkout does not match the pinned revision")
    script = root / "scripts/training/run_recipe.py"
    if not script.is_file():
        parser.error("upstream recipe launcher is missing")
    overrides = [
        "checkpoint.save=null",
        "checkpoint.load=null",
        "checkpoint.pretrained_checkpoint=null",
        "validation.eval_iters=0",
        "logger.log_interval=1",
        f"logger.tensorboard_dir={output / 'tensorboard'}",
        "dataset.num_workers=0",
        "optimizer.adam_beta1=0.9",
        "optimizer.adam_beta2=0.999",
        "optimizer.adam_eps=1e-8",
        "optimizer.weight_decay=0.01",
        "optimizer.clip_grad=0.0",
        "model.moe_router_force_load_balancing=false",
        "ddp.grad_reduce_in_fp32=true",
        "mixed_precision.grad_reduce_in_fp32=true",
        f"model.sequence_parallel={'true' if args.tensor_parallel > 1 else 'false'}",
    ]
    if not args.recompute:
        overrides.extend(
            [
                "model.recompute_granularity=null",
                "model.recompute_method=null",
                "model.recompute_num_layers=null",
            ]
        )
    command = [
        str(args.python.resolve()),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.world_size}",
        "--max-restarts=0",
        str(script),
        "--recipe",
        RECIPE,
        "--mode",
        "pretrain",
        "--dataset",
        "mock",
        "--seq_length",
        str(args.sequence_length),
        "--global_batch_size",
        str(args.global_tokens // args.sequence_length),
        "--micro_batch_size",
        str(args.microbatch_tokens // args.sequence_length),
        "--max_steps",
        str(args.warmup_steps + args.steps),
        "--warmup_iters",
        "0",
        "--lr",
        "0.0003",
        "--min_lr",
        "0.0003",
        "--tensor_model_parallel_size",
        str(args.tensor_parallel),
        "--pipeline_model_parallel_size",
        str(args.pipeline_parallel),
        "--expert_model_parallel_size",
        str(args.expert_parallel),
        "--expert_tensor_parallel_size",
        str(args.expert_tensor_parallel),
        *overrides,
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "request.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "bridge_revision": BRIDGE_REVISION,
            "core_revision": CORE_REVISION,
            "recipe": RECIPE,
            "data_parallel_size": dp,
            "gradient_accumulation": accumulation,
            "command": command,
            "warmup_note": (
                "Exclude the first warmup_steps complete updates in analysis; "
                "warmup_iters=0 disables LR warmup only"
            ),
            "status": "prepared_not_gpu_validated",
        },
    )
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    launcher = output / "run.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + f"cd {shlex.quote(str(output))}\n"
        + "exec "
        + shlex.join(command)
        + "\n"
    )
    launcher.chmod(0o755)
    print(launcher)


if __name__ == "__main__":
    main()
