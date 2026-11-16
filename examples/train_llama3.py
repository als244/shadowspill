"""Llama training from a lightweight client connected to an existing service."""

import argparse
from dataclasses import fields, replace
from pathlib import Path

from examples.training.llama3.loop import train
from examples.training.llama3.setup import Settings, read_resume, start
from examples.training.text_data import TokenBatches
from shadowspill_client import SSH, ServiceClient


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--server", help="Existing HTTP(S) service URL")
    connection.add_argument("--ssh", help="SSH host alias of an existing service")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Saved checkpoint reference JSON")
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
    requested = {
        field.name: vars(args)[field.name]
        for field in fields(Settings)
        if field.name in vars(args)
    }
    if args.resume is None:
        settings = Settings(**requested)
        resume = None
    else:
        resume, settings = read_resume(args.resume)
        if replace(settings, **requested) != settings:
            raise ValueError("resume requires the saved training settings and schedule")
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
    root = args.run_dir.resolve()
    root.mkdir(parents=True)  # Preserve all prior local runs.
    connection = (
        {"ssh": SSH(args.ssh, args.server_port)}
        if args.ssh
        else {"base_url": args.server or f"http://127.0.0.1:{args.server_port}"}
    )
    with ServiceClient(**connection) as service:
        training, cursor = start(service, settings, source, root, resume=resume)
        try:
            train(training, settings, source, validation, root, cursor=cursor)
        except BaseException as error:
            # Keep the original failure and the saved reconnect IDs. No update
            # is retried; a transport-uncertain handle may also reject close.
            try:
                training.close().result()
            except BaseException as cleanup:
                error.add_note(f"Training close also failed: {cleanup}")
            raise
        else:
            training.close().result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
