"""CLI for independent recipe phases; execution settings stay in config.json."""

import argparse


def parser():
    result = argparse.ArgumentParser(
        description="GLM-5.3-Flash checkpoint LoRA through saved phases"
    )
    result.add_argument(
        "--data", help="Corpus prepared with this checkpoint's tokenizer"
    )
    result.add_argument("--checkpoint", help="Local Hugging Face base checkpoint")
    result.add_argument("--frozen-state-id", help="Immutable base checkpoint identity")
    result.add_argument("--run-dir", required=True)
    result.add_argument(
        "--phase",
        choices=["all", "build", "profile", "communicate", "plan", "execute"],
        default="all",
    )
    result.add_argument("--ranks", type=int, default=1)
    result.add_argument(
        "--resume", help="Checkpoint archive; {rank:05d} selects each rank's file"
    )
    result.add_argument("--sequence-length", type=int, default=32768)
    result.add_argument("--sequences-per-step", type=int, default=1, help="Per rank")
    result.add_argument("--max-sequences-per-example", type=int, default=1)
    result.add_argument(
        "--steps",
        type=int,
        help="Total updates, including restored progress; default 5",
    )
    result.add_argument("--schedule-steps", type=int, default=10)
    result.add_argument("--lora-rank", type=int, default=32)
    result.add_argument("--lora-alpha", type=float, default=32)
    result.add_argument(
        "--router-weight-dtype",
        choices=["bf16", "fp32"],
        help="Optional checkpoint router storage override",
    )
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--head-chunk-size", type=int, default=1024)
    result.add_argument(
        "--layer-limit", type=int, help="Debug prefix; omit for full model"
    )
    result.add_argument(
        "--execution-gib", type=float, help="Total GPU budget; default 76"
    )
    result.add_argument(
        "--spill-gib", type=float, help="Per-rank spill budget; default 96"
    )
    result.add_argument("--external-headroom-mib", type=int, default=2048)
    result.add_argument("--eval-every", type=int, default=2)
    result.add_argument("--eval-updates", type=int, default=1)
    result.add_argument("--checkpoint-every", type=int, default=2)
    result.add_argument("--search-workers", type=int, default=8)
    result.add_argument(
        "--resolution-options",
        nargs="+",
        default=["1"],
        help="Allowed fractions of stages to recompute",
    )
    result.add_argument("--warmup-seconds", type=float, default=0.05)
    result.add_argument("--measure-seconds", type=float, default=0.18)
    result.add_argument("--no-plots", dest="plots", action="store_false")
    result.add_argument(
        "--nsys-steps",
        type=int,
        help="Updates captured under cudaProfilerApi; default disabled",
    )
    result.add_argument(
        "--nsys-start-step", type=int, help="First captured update; default 2"
    )
    result.add_argument("--repeat-training-batch", action="store_true")
    result.add_argument("--minimum-relative-loss-drop", type=float)
    result.add_argument("--seed", type=int, default=0)
    return result
