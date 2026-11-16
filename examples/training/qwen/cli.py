"""Qwen training from a lightweight client connected to an existing service."""

import argparse

from examples.training.application.cli import (
    argument_parser,
    connect,
    run_directory,
    settings_from_args,
)
from examples.training.application.loop import train
from examples.training.application.records import data_identity, record
from examples.training.application.setup import close_after, start
from examples.training.qwen.settings import Settings, qwen3_settings
from examples.training.qwen.setup import resolve
from examples.training.text_data import TokenBatches, validate_vocabulary


def arguments(argv=None, *, family="qwen35moe"):
    parser = argument_parser(__doc__)
    parser.add_argument("--wandb-project", help="Optional caller-side W&B logging")
    for name in ("data", "checkpoint", "compute-dtype"):
        parser.add_argument("--" + name, default=argparse.SUPPRESS)
    if family == "qwen3moe":
        parser.add_argument(
            "--preset",
            default=argparse.SUPPRESS,
            help="Installed random-weight model preset; default: 30b-a3b",
        )
        parser.add_argument(
            "--parallelism", choices=("dp", "ep"), default=argparse.SUPPRESS
        )
    for name in (
        "sequence-length",
        "sequences-per-step",
        "steps",
        "schedule-steps",
        "ranks",
        "backward-pieces",
        "seed",
        "eval-every",
        "eval-updates",
        "checkpoint-every",
        "search-workers",
        "external-headroom-mib",
    ):
        parser.add_argument("--" + name, type=int, default=argparse.SUPPRESS)
    for name in (
        "learning-rate",
        "balance-weight",
        "execution-gib",
        "spill-gib",
        "warmup-seconds",
        "measure-seconds",
    ):
        parser.add_argument("--" + name, type=float, default=argparse.SUPPRESS)
    if family == "qwen35moe":
        parser.add_argument("--lora-rank", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--lora-alpha", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--reject-overbudget",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    for name in (
        "microbatch-sizes",
        "chunk-sizes",
        "moe-chunk-sizes",
        "tail-save-counts",
    ):
        parser.add_argument("--" + name, nargs="+", type=int, default=argparse.SUPPRESS)
    for name in ("graph-pair-policies", "resolution-options"):
        parser.add_argument("--" + name, nargs="+", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.resume is None:
        required = ("data", "checkpoint") if family == "qwen35moe" else ("data",)
        if not all(getattr(args, n, None) for n in required):
            parser.error(
                "fresh training requires " + " and ".join("--" + n for n in required)
            )
    return args


def main(argv=None, *, family="qwen35moe"):
    args = arguments(argv, family=family)
    settings, resume = settings_from_args(
        args,
        Settings,
        create=qwen3_settings if family == "qwen3moe" else Settings,
        application=(
            "shadowspill.example.qwen3_full@1"
            if family == "qwen3moe"
            else "shadowspill.example.qwen35_lora@1"
        ),
    )
    source = TokenBatches(
        settings.data,
        settings.sequence_length,
        settings.sequences_per_step,
        normalize=False,
    )
    validation = (
        TokenBatches(
            settings.data,
            settings.sequence_length,
            settings.sequences_per_step,
            split="val",
            normalize=False,
        )
        if settings.eval_every
        else None
    )
    if settings.steps > len(source) or (
        validation is not None and settings.eval_updates > len(validation)
    ):
        raise ValueError("corpus does not contain the requested complete updates")
    root = run_directory(args)
    logger = None
    if args.wandb_project:
        import wandb

        logger = wandb.init(
            project=args.wandb_project, config=record(settings), dir=str(root)
        )
    try:
        with connect(args) as service:
            preview = None
            if resume is None:
                preview = resolve(service, settings)
                validate_vocabulary(preview, source, validation)
            training, cursor = start(
                service,
                settings,
                root,
                data=data_identity(source, validation),
                resume=resume,
                preview=preview,
            )
            with close_after(training):
                train(
                    training,
                    settings,
                    source,
                    validation,
                    root,
                    cursor=cursor,
                    logger=logger,
                )
    finally:
        if logger is not None:
            logger.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
