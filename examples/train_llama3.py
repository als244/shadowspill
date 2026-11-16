"""Llama training from a lightweight client connected to an existing service."""

import argparse

from examples.training.application.cli import (
    argument_parser,
    connect,
    run_directory,
    settings_from_args,
)
from examples.training.application.setup import close_after, start
from examples.training.llama3.loop import train
from examples.training.llama3.setup import Settings
from examples.training.text_data import TokenBatches, validate_vocabulary


def arguments(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--data",
        default=argparse.SUPPRESS,
        help="Local uint32 train/val.bin and meta.json corpus",
    )
    parser.add_argument(
        "--model",
        choices=["llama3-tiny", "llama3-1b", "llama3-8b"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--sequence-length", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sequences-per-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sequences-per-example", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--execution-gib", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--spill-gib", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--warmup-seconds", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--measure-seconds", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--backward-pieces",
        type=int,
        default=argparse.SUPPRESS,
        help="Unequal Datum groups accumulated before each optimizer call",
    )
    args = parser.parse_args(argv)
    if args.resume is None and not getattr(args, "data", None):
        parser.error("fresh training requires --data")
    return args


def main(argv=None):
    args = arguments(argv)
    settings, resume = settings_from_args(args, Settings)
    source = TokenBatches(
        settings.data, settings.sequence_length, settings.sequences_per_step
    )
    validation = TokenBatches(
        settings.data,
        settings.sequence_length,
        settings.sequences_per_step,
        split="val",
    )
    if settings.steps > len(source):
        raise ValueError("corpus does not contain the requested number of updates")
    root = run_directory(args)
    with connect(args) as service:
        preview = None
        if resume is None:
            preview = service.describe_training(**settings.request()).result()
            validate_vocabulary(preview, source)
        training, cursor = start(
            service,
            settings,
            root,
            data=source.identity,
            resume=resume,
            preview=preview,
        )
        with close_after(training):
            train(training, settings, source, validation, root, cursor=cursor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
