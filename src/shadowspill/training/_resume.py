"""Trainer progress composed with the bounded Executor checkpoint helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shadowspill.checkpoint import load_checkpoint, save_checkpoint
from shadowspill.checkpoint.load import checkpoint_metadata
from shadowspill.pytorch._rng import restore_rng, rng_state

from . import _state

if TYPE_CHECKING:
    from .trainer import Trainer


def save(trainer: Trainer, path: str | Path, *, source: Any, chunk_bytes: int) -> Path:
    source = trainer._source if source is None else source
    source_state = getattr(source, "state_dict", None)
    schedules = {}
    for name, schedule in trainer.schedules.items():
        snapshot = getattr(schedule, "state_dict", None)
        if callable(snapshot):
            schedules[name] = snapshot()
    trainer.executor.synchronize()
    metadata = {
        "format": "shadowspill.trainer",
        "version": 1,
        "step": trainer.step_count,
        "elapsed_seconds": trainer.elapsed_seconds,
        "schedule_names": sorted(trainer.schedules),
        "schedules": _state.encode(schedules),
        "source": _state.encode(
            source_state() if callable(source_state) else trainer._source_state
        ),
        "rng": _state.encode(rng_state(trainer.executor.device)),
    }
    save_checkpoint(trainer.executor, path, metadata=metadata, chunk_bytes=chunk_bytes)
    return Path(path)


def load(trainer: Trainer, path: str | Path, *, chunk_bytes: int) -> None:
    # Validate loop compatibility before changing persistent tensors. File-level
    # validation is shared with the low-level checkpoint loader.
    metadata = checkpoint_metadata(trainer.executor, path)
    if metadata.get("format") != "shadowspill.trainer" or metadata.get("version") != 1:
        raise ValueError("checkpoint has no compatible Trainer progress")
    if metadata.get("schedule_names") != sorted(trainer.schedules):
        raise ValueError("checkpoint schedules differ from Trainer schedules")
    step, elapsed = metadata["step"], metadata["elapsed_seconds"]
    if (
        type(step) is not int
        or step < 0
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
    ):
        raise ValueError("invalid Trainer progress")
    schedules, source, rng = (
        _state.decode(metadata[k]) for k in ("schedules", "source", "rng")
    )
    for name in schedules:
        if not callable(getattr(trainer.schedules.get(name), "load_state_dict", None)):
            raise ValueError(f"schedule {name!r} cannot restore saved state")
    restored = load_checkpoint(trainer.executor, path, chunk_bytes=chunk_bytes)
    if restored != metadata:
        raise ValueError("checkpoint metadata changed during restore")
    for name, state in schedules.items():
        restore = getattr(trainer.schedules[name], "load_state_dict", None)
        assert callable(restore)
        restore(state)
    trainer.step_count = step
    trainer._elapsed_seconds = elapsed
    trainer._started = time.perf_counter()
    trainer._source_state = source
    restore_rng(rng, trainer.executor.device)
