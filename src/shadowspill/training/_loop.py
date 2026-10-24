"""The optional data loop; scheduling and memory management belong to Executor."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _logs
from .results import StepResult

if TYPE_CHECKING:
    from .trainer import Trainer


def fit(
    trainer: Trainer,
    source: Iterable[Mapping[str, Any]],
    *,
    steps: int,
    run_dir: str | Path | None,
    log_every: int,
    tables_every: int,
    logger: Callable[[dict[str, Any]], None] | None,
    callbacks: Sequence[Callable[[Trainer, StepResult], None]],
    checkpoint_every: int,
    checkpoint_dir: str | Path | None,
    keep_last: int,
) -> StepResult | None:
    trainer._require_open()
    if type(steps) is not int or steps < trainer.step_count:
        raise ValueError("steps is the target total completed update count")
    if any(
        type(v) is not int or v < 0 for v in (log_every, tables_every, checkpoint_every)
    ):
        raise ValueError("cadences must be nonnegative integers")
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be a positive integer")
    if checkpoint_every and checkpoint_dir is None and run_dir is None:
        raise ValueError("checkpointing needs checkpoint_dir or run_dir")
    trainer._source = source
    if trainer._source_state is not None:
        restore = getattr(source, "load_state_dict", None)
        if not callable(restore):
            raise ValueError(
                "checkpoint has source progress; supply a resumable source"
            )
        restore(trainer._source_state)
        trainer._source_state = None
    if steps == trainer.step_count:
        return None
    iterator = iter(source)
    root = None if run_dir is None else Path(run_dir)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    output = (
        nullcontext(None)
        if root is None
        else (root / "metrics.jsonl").open("a", buffering=1)
    )
    last = None
    with output as stream:
        while trainer.step_count < steps:
            try:
                data = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    f"data source ended after {trainer.step_count} of {steps} updates"
                ) from error
            if not isinstance(data, Mapping):
                raise TypeError(
                    "each source item must contain keyword arguments for Trainer.step"
                )
            last = trainer.step(**data)
            if _logs.due(log_every, last.step, steps):
                _logs.emit(last, trainer.elapsed_seconds, stream, logger)
            if _logs.due(tables_every, last.step, steps):
                _logs.tables(last, root, logger)
            # Callbacks run before checkpointing so saved application/source
            # progress includes changes associated with the completed update.
            for callback in callbacks:
                callback(trainer, last)
            if _logs.due(checkpoint_every, last.step, steps):
                if checkpoint_dir is None:
                    assert root is not None
                    directory = root / "checkpoints"
                else:
                    directory = Path(checkpoint_dir)
                trainer.save(directory / f"step_{last.step:08d}.ssp")
                _logs.prune(directory, keep_last)
    return last
