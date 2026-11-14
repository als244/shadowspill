"""Host-side logging and retention for completed updates."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .results import StepResult


def due(every: int, done: int, target: int) -> bool:
    return every > 0 and (done % every == 0 or done == target)


def emit(
    result: StepResult,
    elapsed: float,
    stream: TextIO | None,
    logger: Callable[[dict[str, Any]], None] | None,
) -> None:
    record = {
        "step": result.step,
        "train/loss": result.loss,
        "train/step_seconds": result.seconds,
        "train/elapsed_seconds": elapsed,
        **{"hyperparameters/" + k: _value(v) for k, v in result.hyperparams.items()},
        **{"train/" + k: _value(v) for k, v in result.summary.scalars.items()},
    }
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if stream is not None:
        stream.write(line + "\n")
    if logger is not None:
        logger(record)


def _value(value: Any) -> Any:
    # Hyperparameters have already been copied to CPU at the step boundary.
    return value.tolist() if hasattr(value, "tolist") else value


def tables(
    result: StepResult, root: Path | None, logger: Any, *, phase: str = "train"
) -> None:
    for name, table in result.summary.tables.items():
        key = phase + "/" + name
        if root is not None:
            with (root / "observations.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "step": result.step,
                            "name": key,
                            "columns": table.columns,
                            "rows": table.rows,
                        }
                    )
                    + "\n"
                )
        emit_table = getattr(logger, "table", None)
        if callable(emit_table):
            emit_table(result.step, key, table)


def prune(directory: Path, keep_last: int) -> None:
    checkpoints = sorted(directory.glob("step_????????.ssp"))
    for path in checkpoints[:-keep_last]:
        if path.is_file():
            path.unlink()
