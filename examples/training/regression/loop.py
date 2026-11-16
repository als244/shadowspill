"""Submit global batches to a ready client and retain completed update records."""

import json
from dataclasses import asdict

from .setup import APPLICATION


def train(training, settings, source, directory, *, cursor=0):
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    with (directory / "metrics.jsonl").open("x", buffering=1) as metrics:
        for index in range(cursor, settings.steps):
            batch = source[index]
            if settings.backward_pieces == 1:
                result = training.step(batch).result()
                loss = result.loss
            else:
                loss = 0.0
                for part in range(settings.backward_pieces):
                    start = part * len(batch) // settings.backward_pieces
                    end = (part + 1) * len(batch) // settings.backward_pieces
                    loss += training.forward_backward(batch[start:end]).result().loss
                result = training.optim_step().result()
            if result.outcome != "completed" or result.completed_updates != index + 1:
                raise RuntimeError("training did not complete the expected update")
            record = {
                "step": result.completed_updates,
                "loss": loss,
                "samples": len(batch),
                "request_id": result.request_id,
            }
            metrics.write(json.dumps(record) + "\n")
            print(record, flush=True)
            if settings.checkpoint_every and (
                result.completed_updates % settings.checkpoint_every == 0
                or result.completed_updates == settings.steps
            ):
                checkpoint = training.save_checkpoint(
                    name=f"step-{result.completed_updates:08d}",
                    metadata={
                        "application": APPLICATION,
                        "settings": asdict(settings),
                        "data": source.identity,
                        "next_update": result.completed_updates,
                    },
                ).result()
                record = {
                    "application": APPLICATION,
                    "settings": asdict(settings),
                    "checkpoint": checkpoint.reference.to_dict(),
                    "plans": training.plans.reference.to_dict(),
                }
                (
                    checkpoint_dir / f"step-{result.completed_updates:08d}.json"
                ).write_text(json.dumps(record, indent=2) + "\n")
    training.report().result().download(directory / "training.html").result()


def export_wandb(directory, *, project, mode):
    """Optional reporting cannot repeat or invalidate a completed update."""
    import wandb

    info = json.loads((directory / "run.json").read_text())
    with wandb.init(
        project=project, mode=mode, dir=directory, config=info["settings"]
    ) as run, (directory / "metrics.jsonl").open() as records:
        for line in records:
            row = json.loads(line)
            run.log(
                {"train/loss": row["loss"], "train/samples": row["samples"]},
                step=row["step"],
            )
