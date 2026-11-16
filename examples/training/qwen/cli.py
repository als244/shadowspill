"""Qwen training from a lightweight client connected to an existing service."""

import argparse

from examples.training.application.cli import (
    argument_parser,
    settings_from_args,
)
from examples.training.qwen.settings import Settings, qwen3_settings
from examples.training.qwen.setup import resolve
from examples.training.text.cli import add_training_arguments, run


def arguments(argv=None, *, family="qwen35moe"):
    parser = argument_parser(__doc__)
    add_training_arguments(parser)
    for name in ("checkpoint", "compute-dtype"):
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
    parser.add_argument("--balance-weight", type=float, default=argparse.SUPPRESS)
    if family == "qwen35moe":
        parser.add_argument("--lora-rank", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--lora-alpha", type=float, default=argparse.SUPPRESS)
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
    return run(args, settings, resume=resume, describe=resolve)


if __name__ == "__main__":
    raise SystemExit(main())
