"""Submit inputs and consume results using an already initialized client."""

import json
from time import monotonic

from shadowspill_client.display import CommandTable
from shadowspill_client.schedules import WarmupCosine
from shadowspill_client.text import text_batch_statistics

from .records import data_identity, save_checkpoint


def command(
    training, operation, batch=None, *, table=None, statistics=None, **controls
):
    """Track one submitted compute command; timing comes from its server result."""
    planned = facts = None
    if table is not None and batch is not None:
        validation = training.validate_batch(
            batch, operation=operation, outputs=controls.get("outputs")
        ).result()
        planned = validation.estimated_seconds
        facts = None if statistics is None else statistics(batch)
    submit = getattr(training, operation)
    pending = submit(**controls) if batch is None else submit(batch, **controls)
    if table is not None:
        table.track(operation, pending, batch=facts, planned_seconds=planned)
    return pending.result()


def update(
    training, batch, *, pieces, optim=None, grad_scale=None, table=None, statistics=None
):
    """One public update, with optional caller controls and final normalization."""
    controls = {}
    if optim is not None:
        controls["optim"] = optim
    if grad_scale is not None:
        controls["grad_scale"] = grad_scale
    scale = 1 if grad_scale is None else grad_scale
    if pieces == 1:
        result = command(
            training, "step", batch, table=table, statistics=statistics, **controls
        )
        return result, result.loss * scale, result.execution_seconds
    total = 0.0
    durations = []
    for part in range(pieces):
        start, stop = part * len(batch) // pieces, (part + 1) * len(batch) // pieces
        contribution = command(
            training,
            "forward_backward",
            batch[start:stop],
            table=table,
            statistics=statistics,
        )
        total += contribution.loss
        durations.append(contribution.execution_seconds)
    result = command(training, "optim_step", table=table, **controls)
    durations.append(result.execution_seconds)
    seconds = None if any(value is None for value in durations) else sum(durations)
    return result, total * scale, seconds


def train(training, settings, source, validation, directory, *, cursor=0, logger=None):
    schedule = WarmupCosine(
        settings.learning_rate, settings.learning_rate / 10, 1, settings.schedule_steps
    )
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    table = CommandTable()
    with (
        (directory / "metrics.jsonl").open("x", buffering=1) as metrics,
        (directory / "evaluation.jsonl").open("x", buffering=1) as evaluations,
    ):
        for index in range(cursor, settings.steps):
            started = monotonic()
            result, loss, seconds = update(
                training,
                source[index],
                pieces=settings.backward_pieces,
                optim={"lr": schedule(index)},
                grad_scale=1 / source.positions,
                table=table,
                statistics=text_batch_statistics,
            )
            if result.outcome != "completed" or result.completed_updates != index + 1:
                raise RuntimeError("training did not complete the expected update")
            step = result.completed_updates
            row = {
                "step": step,
                "loss": loss,
                "lr": schedule(index),
                "execution_seconds": seconds,
                "client_elapsed_seconds": monotonic() - started,
                "request_id": result.request_id,
            }
            metrics.write(json.dumps(row) + "\n")
            scalars = {"train/loss": loss, "train/lr": schedule(index)}
            if settings.eval_every and (
                step % settings.eval_every == 0 or step == settings.steps
            ):
                total = sum(
                    command(
                        training,
                        "forward",
                        validation[n],
                        outputs="metrics",
                        table=table,
                        statistics=text_batch_statistics,
                    ).loss
                    for n in range(settings.eval_updates)
                )
                value = total / (settings.eval_updates * validation.positions)
                row = {"step": step, "validation_loss": value}
                evaluations.write(json.dumps(row) + "\n")
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
