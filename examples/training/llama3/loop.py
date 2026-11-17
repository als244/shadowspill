"""Only inputs, training settings and results; the client is already ready."""

import json
from time import monotonic

from examples.training.application.loop import command, update
from examples.training.application.records import save_checkpoint
from shadowspill_client.display import CommandTable
from shadowspill_client.schedules import WarmupCosine
from shadowspill_client.text import text_batch_statistics

from .settings import APPLICATION


def train(training, settings, source, validation, directory, *, cursor=0):
    schedule = WarmupCosine(
        settings.learning_rate,
        settings.learning_rate / 10,
        min(1, settings.steps - 1),
        settings.steps,
    )
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    table = CommandTable()
    with (
        (directory / "metrics.jsonl").open("x", buffering=1) as metrics,
        (directory / "evaluation.jsonl").open("x", buffering=1) as evaluations,
    ):

        def evaluate(step):
            result = command(
                training,
                "forward",
                validation[0],
                outputs="metrics",
                table=table,
                statistics=text_batch_statistics,
            )
            record = {"step": step, "validation_loss": result.loss}
            evaluations.write(json.dumps(record) + "\n")
            return result.loss

        evaluate(cursor)
        for index in range(cursor, settings.steps):
            started = monotonic()
            result, loss, seconds = update(
                training,
                source[index],
                pieces=settings.backward_pieces,
                optim={"lr": schedule(index)},
                table=table,
                statistics=text_batch_statistics,
            )
            if result.outcome != "completed" or result.completed_updates != index + 1:
                raise RuntimeError("training did not complete the expected update")
            record = {
                "step": result.completed_updates,
                "loss": loss,
                "lr": schedule(index),
                "execution_seconds": seconds,
                "client_elapsed_seconds": monotonic() - started,
                "request_id": result.request_id,
            }
            metrics.write(json.dumps(record) + "\n")
            validation_loss = evaluate(result.completed_updates)
            training.log_metrics(
                {"eval/loss": validation_loss, "train/lr": schedule(index)},
                request_id=result.request_id,
            ).result()
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
