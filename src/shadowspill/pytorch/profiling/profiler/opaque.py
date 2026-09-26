"""Eager optimizer tasks: the update run on a materialized optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    materialize_opaque_optimizer,
)
from shadowspill.task.profiles import TaskMeasurement

from .measurement import MeasuredTask, measure_task

if TYPE_CHECKING:
    from . import TaskProfiler


def measure_opaque_optimizer(
    profiler: TaskProfiler,
    artifact: OpaqueOptimizerArtifact,
) -> TaskMeasurement:
    """Measure one eager optimizer task on a materialized optimizer."""

    optimizer = materialize_opaque_optimizer(
        artifact, device_ordinal=profiler.boundary.device_ordinal
    )

    def update(profiled: torch.optim.Optimizer = optimizer) -> object:
        with torch.no_grad():
            return profiled.step()

    try:
        return measure_task(
            profiler, MeasuredTask(update), execution_provider="opaque-optimizer"
        )
    finally:
        del update
        del optimizer
