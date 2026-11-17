"""Submit global batches to a ready client and retain completed update records."""

import json
from time import monotonic

from examples.training.application.loop import update
from examples.training.application.records import save_checkpoint
from shadowspill_client.display import CommandTable

from .settings import APPLICATION


def train(training, settings, source, directory, *, cursor=0):
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    table = CommandTable()
    with (directory / "metrics.jsonl").open("x", buffering=1) as metrics:
        for index in range(cursor, settings.steps):
            started = monotonic()
            batch = source[index]
            result, loss, seconds = update(
                training, batch, pieces=settings.backward_pieces, table=table
            )
            if result.outcome != "completed" or result.completed_updates != index + 1:
                raise RuntimeError("training did not complete the expected update")
            record = {
                "step": result.completed_updates,
                "loss": loss,
                "samples": len(batch),
                "execution_seconds": seconds,
                "client_elapsed_seconds": monotonic() - started,
                "request_id": result.request_id,
            }
            metrics.write(json.dumps(record) + "\n")
            if settings.checkpoint_every and (
                result.completed_updates % settings.checkpoint_every == 0
                or result.completed_updates == settings.steps
            ):
                save_checkpoint(
                    training,
                    checkpoint_dir,
                    step=result.completed_updates,
                    application=APPLICATION,
                    settings=settings,
                    data=source.identity,
                )
    training.report().result().download(directory / "training.html").result()


def export_wandb(directory, *, project, mode):
    """Optional reporting cannot repeat or invalidate a completed update."""
    import wandb

    info = json.loads((directory / "run.json").read_text())
    with (
        wandb.init(
            project=project, mode=mode, dir=directory, config=info["settings"]
        ) as run,
        (directory / "metrics.jsonl").open() as records,
    ):
        for line in records:
            row = json.loads(line)
            run.log(
                {"train/loss": row["loss"], "train/samples": row["samples"]},
                step=row["step"],
            )
