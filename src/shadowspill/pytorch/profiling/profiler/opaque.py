"""Eager optimizer tasks: the recurrent update, and the state-creating first step."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    materialize_opaque_optimizer,
    opaque_optimizer_outputs,
)

from ..records import TaskMeasurement
from .measurement import MeasuredTask, measure_task

if TYPE_CHECKING:
    from . import TaskProfiler


def measure_opaque_optimizer(
    profiler: TaskProfiler,
    artifact: OpaqueOptimizerArtifact,
) -> TaskMeasurement:
    """Measure one eager optimizer task on a materialized optimizer."""

    if artifact.profile_output_names:
        return measure_task(
            profiler,
            FirstStep(profiler, artifact),
            execution_provider="opaque-optimizer-initial",
        )
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


class FirstStep(MeasuredTask):
    """A state-creating first step, built fresh for every invocation.

    Reusing one optimizer would turn every invocation after the first into
    the recurrent update and silently omit lazy-state allocations.  Each
    sample therefore materializes the same storage-free optimizer before the
    measured boundary opens, then destroys it once the invocation's output
    allocations have been classified.  No callable outlives the measurement,
    so the record describes one that refuses to run.
    """

    def __init__(
        self, profiler: TaskProfiler, artifact: OpaqueOptimizerArtifact
    ) -> None:
        super().__init__(_not_retained)
        self.profiler = profiler
        self.artifact = artifact

    @contextmanager
    def open(self) -> Iterator[Callable[[], object]]:
        boundary = self.profiler.boundary
        device_ordinal = boundary.device_ordinal
        optimizer = materialize_opaque_optimizer(
            self.artifact, device_ordinal=device_ordinal
        )
        update = _first_step(self.artifact, optimizer, device_ordinal)
        try:
            yield update
        finally:
            del update
            del optimizer
            # The invocation drained before it returned, but the optimizer it
            # built and the outputs it produced are released as this frame
            # closes, which is after that drain. Their retirements are
            # event-fenced, so reading live bytes now would count memory that
            # is already free -- a few hundred bytes that move with how far
            # the worker happened to get, which no equality test can see
            # through.
            boundary.drain(
                torch.cuda.current_stream(device_ordinal),
                problem="opaque optimizer first step",
            )


def _first_step(
    artifact: OpaqueOptimizerArtifact,
    optimizer: torch.optim.Optimizer,
    device_ordinal: int,
) -> Callable[[], object]:
    """One update on a fresh optimizer, returning the tensors it created."""

    def update() -> object:
        with torch.no_grad():
            optimizer.step()
        return tuple(
            binding.tensor
            for binding in opaque_optimizer_outputs(
                artifact, optimizer, device_ordinal=device_ordinal
            )
        )

    return update


def _not_retained() -> object:
    raise AssertionError("first-step profile callable is not retained")
