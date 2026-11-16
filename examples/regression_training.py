"""Train a small regression model through an existing local or remote service."""

import argparse
import warnings
from dataclasses import fields, replace
from pathlib import Path

from examples.training.regression.data import SyntheticBatches
from examples.training.regression.loop import export_wandb, train
from examples.training.regression.setup import Settings, read_resume, start
from shadowspill_client import SSH, ServiceClient


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--server", help="Existing HTTP(S) service URL")
    connection.add_argument("--ssh", help="SSH host alias of an existing service")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Saved checkpoint reference JSON")
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
    requested = {
        field.name: vars(args)[field.name]
        for field in fields(Settings)
        if field.name in vars(args)
    }
    if args.resume is None:
        settings, resume = Settings(**requested), None
    else:
        resume, settings = read_resume(args.resume)
        if replace(settings, **requested) != settings:
            raise ValueError("resume requires the saved training settings")
    source = SyntheticBatches(
        ranks=settings.ranks,
        samples_per_rank=settings.samples_per_rank,
        seed=settings.seed,
    )
    root = args.run_dir.resolve()
    root.mkdir(parents=True)
    connection = (
        {"ssh": SSH(args.ssh, args.server_port)}
        if args.ssh
        else {"base_url": args.server or f"http://127.0.0.1:{args.server_port}"}
    )
    with ServiceClient(**connection) as service:
        training, cursor = start(service, settings, source, root, resume=resume)
        try:
            train(training, settings, source, root, cursor=cursor)
        except BaseException as error:
            try:
                training.close().result()
            except BaseException as cleanup:
                error.add_note(f"Training close also failed: {cleanup}")
            raise
        else:
            training.close().result()
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
