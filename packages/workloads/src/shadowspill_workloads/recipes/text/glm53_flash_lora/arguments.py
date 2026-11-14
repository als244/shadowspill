"""CLI for independent recipe phases; execution settings stay in config.json."""

import argparse

from shadowspill.training.preparation.configuration import DEFAULTS as RUNNER_DEFAULTS
from shadowspill_workloads.recipes.text.training import DEFAULTS as TEXT_DEFAULTS

from .settings import DEFAULTS


def parser():
    result = argparse.ArgumentParser(
        description="GLM-5.3-Flash checkpoint LoRA through saved phases"
    )
    result.add_argument(
        "--data", help="Corpus prepared with this checkpoint's tokenizer"
    )
    result.add_argument("--checkpoint", help="Local Hugging Face base checkpoint")
    result.add_argument(
        "--initialization",
        choices=["checkpoint", "random"],
        help="Read pretrained values or generate host values using only metadata",
    )
    result.add_argument("--frozen-state-id", help="Immutable base checkpoint identity")
    result.add_argument("--run-dir", required=True)
    result.add_argument(
        "--phase",
        choices=["all", "build", "profile", "communicate", "plan", "execute"],
        default="all",
    )
    result.add_argument("--ranks", type=int)
    result.add_argument(
        "--resume", help="Checkpoint archive; {rank:05d} selects each rank's file"
    )
    result.add_argument("--sequence-length", type=int)
    result.add_argument("--sequences-per-step", type=int, help="Per rank")
    result.add_argument(
        "--microbatch-sizes",
        type=int,
        nargs="+",
        help="Exact sequences-per-microbatch candidates; each must divide "
        "sequences-per-step; default: 1",
    )
    result.add_argument(
        "--steps",
        type=int,
        help="Total updates, including restored progress; default 5",
    )
    result.add_argument("--schedule-steps", type=int)
    result.add_argument("--lora-rank", type=int)
    result.add_argument("--lora-alpha", type=float)
    result.add_argument(
        "--router-weight-dtype",
        choices=["bf16", "fp32"],
        help="Optional checkpoint router storage override",
    )
    result.add_argument("--learning-rate", type=float)
    result.add_argument("--head-chunk-size", type=int)
    result.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        help="Explicit execution candidates (multiples of32); 0 keeps full regions",
    )
    result.add_argument(
        "--moe-chunk-sizes",
        type=int,
        nargs="+",
        help="Expert chunk candidates crossed with chunk-sizes; 0 uses that chunk size",
    )
    result.add_argument(
        "--graph-pair-policies",
        nargs="+",
        choices=["cost-input", "cost-attention", "attention-input"],
        help="Independent SAVE/RECOMPUTE policies, crossed with each chunk candidate",
    )
    result.add_argument(
        "--layer-limit", type=int, help="Debug prefix; omit for full model"
    )
    result.add_argument(
        "--execution-gib", type=float, help="Total GPU budget; default 76"
    )
    result.add_argument(
        "--spill-gib", type=float, help="Per-rank spill budget; default 96"
    )
    result.add_argument("--external-headroom-mib", type=int)
    result.add_argument("--eval-every", type=int)
    result.add_argument("--eval-updates", type=int)
    result.add_argument("--checkpoint-every", type=int)
    result.add_argument("--search-workers", type=int)
    result.add_argument(
        "--resolution-options",
        nargs="+",
        help="Allowed fractions of stages to recompute",
    )
    result.add_argument("--warmup-seconds", type=float)
    result.add_argument("--measure-seconds", type=float)
    result.add_argument("--refresh-profiles", action=argparse.BooleanOptionalAction)
    result.add_argument("--no-plots", dest="plots", action="store_false")
    result.add_argument(
        "--nsys-steps",
        type=int,
        help="Updates captured under cudaProfilerApi; default disabled",
    )
    result.add_argument(
        "--nsys-start-step",
        type=int,
        help="First captured update; default 2",
    )
    result.add_argument("--repeat-training-batch", action="store_true")
    result.add_argument("--minimum-relative-loss-drop", type=float)
    result.add_argument("--seed", type=int)
    defaults = {**RUNNER_DEFAULTS, **TEXT_DEFAULTS, **DEFAULTS}
    for action in result._actions:
        if action.dest in defaults:
            action.default = defaults[action.dest]
    return result
