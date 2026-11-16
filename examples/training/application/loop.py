"""Submit inputs and consume results using an already initialized client."""

import json
from time import monotonic

from shadowspill_client.schedules import WarmupCosine

from .records import data_identity, record


def update(training, batch, *, pieces, lr, positions):
    # Scale the complete objective once, including secondary balancing. Scaling
    # only the primary token weights would leave the auxiliary term unnormalized.
    scale = 1 / positions
    if pieces == 1:
        result = training.step(batch, optim={"lr": lr}, grad_scale=scale).result()
        return result, result.loss * scale
    total = 0.0
    for part in range(pieces):
        start, stop = part * len(batch) // pieces, (part + 1) * len(batch) // pieces
        total += training.forward_backward(batch[start:stop]).result().loss
    result = training.optim_step(optim={"lr": lr}, grad_scale=scale).result()
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
                lr=schedule(index),
                positions=source.positions,
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
                saved = training.save_checkpoint(
                    name=f"step-{step:08d}",
                    metadata={
                        "application": settings.application,
                        "settings": record(settings),
                        "data": data_identity(source, validation),
                        "next_update": step,
                    },
                ).result()
                (checkpoint_dir / f"step-{step:08d}.json").write_text(
                    json.dumps(
                        {
                            "application": settings.application,
                            "settings": record(settings),
                            "checkpoint": saved.reference.to_dict(),
                            "plans": training.plans.reference.to_dict(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
    training.report().result().download(directory / "training.html").result()
