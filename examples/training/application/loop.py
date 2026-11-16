"""Submit inputs and consume results using an already initialized client."""

import json
from time import monotonic

from shadowspill_client.schedules import WarmupCosine

from .records import data_identity, save_checkpoint


def update(training, batch, *, pieces, optim=None, grad_scale=None):
    """One public update, with optional caller controls and final normalization."""
    controls = {}
    if optim is not None:
        controls["optim"] = optim
    if grad_scale is not None:
        controls["grad_scale"] = grad_scale
    scale = 1 if grad_scale is None else grad_scale
    if pieces == 1:
        result = training.step(batch, **controls).result()
        return result, result.loss * scale
    total = 0.0
    for part in range(pieces):
        start, stop = part * len(batch) // pieces, (part + 1) * len(batch) // pieces
        total += training.forward_backward(batch[start:stop]).result().loss
    result = training.optim_step(**controls).result()
    return result, total * scale


def train(training, settings, source, validation, directory, *, cursor=0, logger=None):
    schedule = WarmupCosine(
        settings.learning_rate, settings.learning_rate / 10, 1, settings.schedule_steps
    )
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    with (
        (directory / "metrics.jsonl").open("x", buffering=1) as metrics,
        (directory / "evaluation.jsonl").open("x", buffering=1) as evaluations,
    ):
        for index in range(cursor, settings.steps):
            started = monotonic()
            result, loss = update(
                training,
                source[index],
                pieces=settings.backward_pieces,
                optim={"lr": schedule(index)},
                grad_scale=1 / source.positions,
            )
            if result.outcome != "completed" or result.completed_updates != index + 1:
                raise RuntimeError("training did not complete the expected update")
            step = result.completed_updates
            row = {
                "step": step,
                "loss": loss,
                "lr": schedule(index),
                "seconds": monotonic() - started,
                "request_id": result.request_id,
            }
            metrics.write(json.dumps(row) + "\n")
            print(row, flush=True)
            scalars = {"train/loss": loss, "train/lr": schedule(index)}
            if settings.eval_every and (
                step % settings.eval_every == 0 or step == settings.steps
            ):
                total = sum(
                    training.forward(validation[n], outputs="metrics").result().loss
                    for n in range(settings.eval_updates)
                )
                value = total / (settings.eval_updates * validation.positions)
                row = {"step": step, "validation_loss": value}
                evaluations.write(json.dumps(row) + "\n")
                print(row, flush=True)
                scalars["eval/loss"] = value
            training.log_metrics(scalars, request_id=result.request_id).result()
            if logger is not None:
                logger.log(scalars, step=step)
            if settings.checkpoint_every and (
                step % settings.checkpoint_every == 0 or step == settings.steps
            ):
                save_checkpoint(
                    training,
                    checkpoint_dir,
                    step=step,
                    application=settings.application,
                    settings=settings,
                    data=data_identity(source, validation),
                )
    training.report().result().download(directory / "training.html").result()
