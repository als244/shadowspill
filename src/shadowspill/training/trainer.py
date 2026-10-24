"""Training loops over a prepared Plan, inside an execution worker."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.execution import Executor, TensorChunk, TrainingResult

from . import _loop, _resume
from .observations import MetricSummary, StepObservations, to_host
from .results import StepResult


class Trainer:
    """Run a saved training Plan; construction never builds or profiles a model.

    Construct in a fresh execution worker, before another CUDA allocator is
    initialized, or supply an existing Executor. Only an Executor created here
    is closed here. ``step`` accepts the same rank-local inputs as Executor.run;
    each ``fit`` source item is a mapping of those keyword arguments.
    """

    def __init__(
        self,
        plan: Plan,
        *,
        executor: Executor | None = None,
        evaluation_plan: Plan | None = None,
        schedules: Mapping[str, Callable[[int], Any]] | None = None,
        metric_reducer: Callable[
            [StepObservations], MetricSummary | Mapping[str, float]
        ]
        | None = None,
        device: int = 0,
        rank_map: Mapping[int, int] | None = None,
        external_headroom_bytes: int | None = None,
    ) -> None:
        if not isinstance(plan, Plan) or plan.program.bindings.get("kind") != "step":
            raise TypeError("Trainer requires a Plan produced from build_step")
        if evaluation_plan is not None and (
            not isinstance(evaluation_plan, Plan)
            or evaluation_plan.program.bindings.get("kind") != "forward"
        ):
            raise TypeError("evaluation_plan must be produced from build_forward")
        self.schedules = dict(schedules or {})
        if any(not callable(fn) for fn in self.schedules.values()):
            raise TypeError("each schedule must be callable(completed_steps)")
        names = {
            binding["name"]
            for binding in plan.program.bindings.get("optimizer_bindings", ())
            if binding["role"] == "hyperparameter"
        }
        if unknown := self.schedules.keys() - names:
            raise ValueError(
                f"schedules name unknown runtime hyperparameters: {sorted(unknown)}"
            )
        if executor is not None and (
            executor.plan.program.digest != plan.program.digest
            or executor.plan.schedule.digest != plan.schedule.digest
            or executor.plan.memory_layout != plan.memory_layout
        ):
            raise ValueError("supplied Executor belongs to a different Plan")
        if executor is not None and evaluation_plan is not None:
            bound = executor.plans.get("evaluation")
            if bound is None or (
                bound.program.digest != evaluation_plan.program.digest
                or bound.schedule.digest != evaluation_plan.schedule.digest
                or bound.memory_layout != evaluation_plan.memory_layout
            ):
                raise ValueError("supplied Executor has a different evaluation Plan")
        self.plan = plan
        self.executor = executor or Executor(
            plan,
            device=device,
            rank_map=rank_map,
            external_headroom_bytes=external_headroom_bytes,
            plans=None if evaluation_plan is None else {"evaluation": evaluation_plan},
        )
        self.evaluation_plan = self.executor.plans.get("evaluation")
        self._owns_executor = executor is None
        self.metric_reducer = metric_reducer
        self.step_count = 0
        self._elapsed_seconds = 0.0
        self._started: float | None = time.perf_counter()
        self._source: Any = None
        self._source_state: Any = None
        self._closed = False

    @property
    def elapsed_seconds(self) -> float:
        active = 0.0 if self._started is None else time.perf_counter() - self._started
        return self._elapsed_seconds + active

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Trainer is closed")

    def initialize(
        self,
        fill_tensor: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
    ) -> Self:
        """Fill model tensors in bounded Executor buffers before the first update."""
        self._require_open()
        if self.step_count:
            raise RuntimeError("use load to resume a started Trainer")
        self.executor.initialize(fill_tensor, chunk_bytes=chunk_bytes)
        return self

    def step(
        self,
        *,
        args: Sequence[tuple[Any, ...]] | None = None,
        kwargs: Sequence[Mapping[str, Any]] | None = None,
        input_loader: Callable[[int], tuple[tuple[Any, ...], Mapping[str, Any]]]
        | None = None,
        hyperparams: Mapping[str, Any] | None = None,
    ) -> StepResult:
        """Complete an update with host observations; schedules are zero-based."""
        self._require_open()
        values = {name: fn(self.step_count) for name, fn in self.schedules.items()}
        values.update(hyperparams or {})
        started = time.perf_counter()
        result = self.executor.run(
            args=args, kwargs=kwargs, input_loader=input_loader, hyperparams=values
        ).result()
        if not isinstance(result, TrainingResult):
            raise TypeError("training Plan returned an invalid result")
        self.step_count += 1
        try:
            host = StepObservations.collect(result.losses, result.metrics)
        finally:
            # Release the Executor's completed handle. StepResult keeps only CPU
            # observations, so fit never retains execution-pool output leases.
            self.executor.synchronize()
        seconds = time.perf_counter() - started
        summary = MetricSummary()
        if self.metric_reducer is not None:
            reduced = self.metric_reducer(host)
            summary = (
                MetricSummary(scalars=dict(reduced))
                if isinstance(reduced, Mapping)
                else reduced
            )
            if not isinstance(summary, MetricSummary):
                raise TypeError("metric_reducer must return a mapping or MetricSummary")
        return StepResult(
            host.losses,
            host.metrics,
            host.parameter_metrics,
            self.step_count,
            seconds,
            to_host(values),
            summary,
        )

    def evaluate(
        self,
        *,
        args: Sequence[tuple[Any, ...]] | None = None,
        kwargs: Sequence[Mapping[str, Any]] | None = None,
        input_loader: Callable[[int], tuple[tuple[Any, ...], Mapping[str, Any]]]
        | None = None,
    ) -> Any:
        """Run the prebuilt forward Plan and return its ordinary output on CPU.

        Evaluation uses current compute weights and buffers; it neither steps
        the optimizer nor advances schedules. Losses/metrics belong in the
        forward computation when needed, so they remain profiled/planned work.
        """
        self._require_open()
        if self.evaluation_plan is None:
            raise RuntimeError("Trainer has no prebuilt evaluation Plan")
        pending = self.executor.run(
            plan="evaluation", args=args, kwargs=kwargs, input_loader=input_loader
        )
        try:
            return to_host(pending.result())
        finally:
            self.executor.synchronize()

    def fit(
        self,
        source: Iterable[Mapping[str, Any]],
        *,
        steps: int,
        run_dir: str | Path | None = None,
        log_every: int = 1,
        tables_every: int = 100,
        logger: Callable[[dict[str, Any]], None] | None = None,
        callbacks: Sequence[Callable[[Trainer, StepResult], None]] = (),
        checkpoint_every: int = 0,
        checkpoint_dir: str | Path | None = None,
        keep_last: int = 3,
    ) -> StepResult | None:
        """Consume call arguments until the total completed count is steps."""
        return _loop.fit(
            self,
            source,
            steps=steps,
            run_dir=run_dir,
            log_every=log_every,
            tables_every=tables_every,
            logger=logger,
            callbacks=callbacks,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
            keep_last=keep_last,
        )

    def save(
        self, path: str | Path, *, source: Any = None, chunk_bytes: int = 64 << 20
    ) -> Path:
        """Save bounded tensors plus loop, source, scheduler and default RNG state."""
        self._require_open()
        return _resume.save(self, path, source=source, chunk_bytes=chunk_bytes)

    def load(self, path: str | Path, *, chunk_bytes: int = 64 << 20) -> Self:
        """Restore tensors now; the next fit restores source progress before reading."""
        self._require_open()
        _resume.load(self, path, chunk_bytes=chunk_bytes)
        return self

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_executor:
            self.executor.close()
        else:
            self.executor.synchronize()
        self._elapsed_seconds = self.elapsed_seconds
        self._started = None
        self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Trainer cleanup also failed: {cleanup}")
