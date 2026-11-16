"""Qwen training from a lightweight client connected to an existing service."""

import argparse
from dataclasses import fields, replace
from pathlib import Path

from examples.training.application.loop import train
from examples.training.qwen.settings import Settings, qwen3_settings
from examples.training.qwen.setup import read_resume, record, start
from examples.training.text_data import TokenBatches
from shadowspill_client import SSH, ServiceClient


def arguments(argv=None, *, family="qwen35moe"):
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--server", help="Existing HTTP(S) service URL")
    connection.add_argument("--ssh", help="SSH host alias of an existing service")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Saved checkpoint reference JSON")
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
    requested = {
        f.name: vars(args)[f.name] for f in fields(Settings) if f.name in vars(args)
    }
    if args.resume is None:
        create_settings = qwen3_settings if family == "qwen3moe" else Settings
        settings, resume = create_settings(**requested), None
    else:
        resume, settings = read_resume(args.resume)
        if settings.family != family or (settings.lora_rank is None) != (
            family == "qwen3moe"
        ):
            raise ValueError("resume record belongs to another application")
        if replace(settings, **requested) != settings:
            raise ValueError("resume requires the saved training settings and schedule")
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
    root = args.run_dir.resolve()
    root.mkdir(parents=True)
    connection = (
        {"ssh": SSH(args.ssh, args.server_port)}
        if args.ssh
        else {"base_url": args.server or f"http://127.0.0.1:{args.server_port}"}
    )
    logger = None
    if args.wandb_project:
        import wandb

        logger = wandb.init(
            project=args.wandb_project, config=record(settings), dir=str(root)
        )
    try:
        with ServiceClient(**connection) as service:
            training, cursor = start(
                service, settings, source, validation, root, resume=resume
            )
            try:
                train(
                    training,
                    settings,
                    source,
                    validation,
                    root,
                    cursor=cursor,
                    logger=logger,
                )
            except BaseException as error:
                try:
                    training.close().result()
                except BaseException as cleanup:
                    error.add_note(f"Training close also failed: {cleanup}")
                raise
            else:
                training.close().result()
    finally:
        if logger is not None:
            logger.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
