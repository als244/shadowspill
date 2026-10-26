"""Reference updates, evaluation and checkpoint cadence using shared log formats."""

import time
from pathlib import Path

from shadowspill.training import _logs
from shadowspill.training.results import StepResult

from ..experiment import inputs, logging
from . import checkpoint


def fit(
    engine, source, data, choice, root, options, reduce_metrics, logger, record_packing
):
    steps = options["steps"]
    cadences = {
        "log": options.get("log_every", 1),
        "tables": options.get("metric_tables_every", 100),
        "evaluate": options.get("eval_every", 0),
        "checkpoint": options.get("checkpoint_every", 0),
    }
    if any(
        type(value) is not int or value < 0 for value in (steps, *cadences.values())
    ):
        raise ValueError("steps and cadences must be nonnegative integers")
    keep_last = options.get("keep_last", 3)
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be a positive integer")
    validation = (
        inputs.validation(data, options, choice) if cadences["evaluate"] else None
    )
    completed, elapsed = (0, 0.0)
    if options.get("resume"):
        completed, elapsed = checkpoint.load(options["resume"], engine, source)
    if steps < completed:
        raise ValueError("steps is the target total completed update count")
    started = time.perf_counter()
    last = None
    with (root / "metrics.jsonl").open("a", buffering=1) as stream:
        for step in range(completed + 1, steps + 1):
            last = engine.step(next(source)["args"], step, reduce_metrics)
            if _logs.due(cadences["log"], step, steps):
                _logs.emit(
                    last, elapsed + time.perf_counter() - started, stream, logger
                )
            if _logs.due(cadences["tables"], step, steps):
                _logs.tables(last, root, logger)
            if _logs.due(cadences["evaluate"], step, steps):
                before = time.perf_counter()
                observed = engine.evaluate(validation)
                summary = reduce_metrics(observed)
                logging.emit(
                    root,
                    logger,
                    {
                        "step": step,
                        "eval/loss": sum(observed.losses),
                        "eval/seconds": time.perf_counter() - before,
                        **{
                            "eval/" + name: value
                            for name, value in summary.scalars.items()
                        },
                    },
                )
                _logs.tables(
                    StepResult(observed.losses, step=step, summary=summary),
                    root,
                    logger,
                    phase="eval",
                )
            record_packing(engine, last)
            if _logs.due(cadences["checkpoint"], step, steps):
                directory = Path(options.get("checkpoint_dir") or root / "checkpoints")
                checkpoint.save(
                    directory / f"step_{step:08d}",
                    engine,
                    source,
                    step,
                    elapsed + time.perf_counter() - started,
                    options,
                )
                checkpoint.prune(directory, keep_last)
    return last
