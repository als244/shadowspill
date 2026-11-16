"""Train a small regression model through an existing local or remote service."""

import argparse
import warnings

from examples.training.application.cli import (
    argument_parser,
    connect,
    run_directory,
    settings_from_args,
)
from examples.training.application.setup import close_after, start
from examples.training.regression.data import SyntheticBatches
from examples.training.regression.loop import export_wandb, train
from examples.training.regression.settings import Settings


def arguments(argv=None):
    parser = argument_parser(__doc__)
    for name in (
        "ranks",
        "steps",
        "samples-per-rank",
        "backward-pieces",
        "checkpoint-every",
        "seed",
    ):
        parser.add_argument(f"--{name}", type=int, default=argparse.SUPPRESS)
    for name in ("execution-gib", "spill-gib", "warmup-seconds", "measure-seconds"):
        parser.add_argument(f"--{name}", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    settings, resume = settings_from_args(args, Settings)
    source = SyntheticBatches(
        ranks=settings.ranks,
        samples_per_rank=settings.samples_per_rank,
        seed=settings.seed,
    )
    root = run_directory(args)
    with connect(args) as service:
        training, cursor = start(
            service, settings, root, data=source.identity, resume=resume
        )
        with close_after(training):
            train(training, settings, source, root, cursor=cursor)
    if args.wandb_project:
        try:
            export_wandb(root, project=args.wandb_project, mode=args.wandb_mode)
        except Exception as error:
            warnings.warn(
                f"Training completed; optional W&B export failed: {error}", stacklevel=1
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
