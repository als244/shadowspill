"""Only inputs, training settings and results; the client is already ready."""

import json
from dataclasses import asdict

from shadowspill_client.schedules import WarmupCosine

from .setup import APPLICATION


def update(training, batch, *, pieces, lr):
    if pieces == 1:
        result = training.step(batch, optim={"lr": lr}).result()
        return result, result.loss
    loss = 0.0
    for part in range(pieces):
        start, end = part * len(batch) // pieces, (part + 1) * len(batch) // pieces
        result = training.forward_backward(batch[start:end]).result()
        loss += result.loss
    return training.optim_step(optim={"lr": lr}).result(), loss


def train(training, settings, source, validation, directory, *, cursor=0):
    schedule = WarmupCosine(
        settings.learning_rate,
        settings.learning_rate / 10,
        min(1, settings.steps - 1),
        settings.steps,
    )
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    with (
        (directory / "metrics.jsonl").open("x", buffering=1) as metrics,
        (directory / "evaluation.jsonl").open("x", buffering=1) as evaluations,
    ):

        def evaluate(step):
            result = training.forward(validation[0], outputs="metrics").result()
            record = {"step": step, "validation_loss": result.loss}
            evaluations.write(json.dumps(record) + "\n")
            print(record, flush=True)
            return result.loss

        evaluate(cursor)
        for index in range(cursor, settings.steps):
            result, loss = update(
                training,
                source[index],
                pieces=settings.backward_pieces,
                lr=schedule(index),
            )
            if result.outcome != "completed" or result.completed_updates != index + 1:
                raise RuntimeError("training did not complete the expected update")
            record = {
                "step": result.completed_updates,
                "loss": loss,
                "lr": schedule(index),
                "request_id": result.request_id,
            }
            metrics.write(json.dumps(record) + "\n")
            print(record, flush=True)
            validation_loss = evaluate(result.completed_updates)
            training.log_metrics(
                {"eval/loss": validation_loss, "train/lr": schedule(index)},
                request_id=result.request_id,
            ).result()
            if settings.checkpoint_every and (
                result.completed_updates % settings.checkpoint_every == 0
                or result.completed_updates == settings.steps
            ):
                saved = training.save_checkpoint(
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
                    "checkpoint": saved.reference.to_dict(),
                    "plans": training.plans.reference.to_dict(),
                }
                (
                    checkpoint_dir / f"step-{result.completed_updates:08d}.json"
                ).write_text(json.dumps(record, indent=2) + "\n")
    training.report().result().download(directory / "training.html").result()
