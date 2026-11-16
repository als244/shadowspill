"""Token-corpus training workflow with caller-selected model configuration."""

import argparse

from examples.training.application.cli import connect, run_directory
from examples.training.application.loop import train
from examples.training.application.records import data_identity, record
from examples.training.application.setup import close_after, start
from examples.training.text_data import TokenBatches, validate_vocabulary


def add_training_arguments(parser):
    parser.add_argument("--data", default=argparse.SUPPRESS)
    parser.add_argument("--wandb-project", help="Optional caller-side W&B logging")
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
        "execution-gib",
        "spill-gib",
        "warmup-seconds",
        "measure-seconds",
    ):
        parser.add_argument("--" + name, type=float, default=argparse.SUPPRESS)
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


def run(args, settings, *, resume=None, describe):
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
                preview = describe(service, settings)
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
