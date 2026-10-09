"""A model-independent update engine and optional training loop."""

from __future__ import annotations

import functools
import json
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from pathlib import Path
from typing import Any, Literal, Self, TextIO, cast

import torch
from torch import nn

from shadowspill.pytorch.distributed import Distributed

from . import _checkpoint
from ._inputs import ScaledObjective, candidate_functions, make_microbatches
from ._model import (
    OptimizerFactory,
    initialize_model,
    model_mode,
    validate_initialization,
)
from ._types import (
    Backend,
    Initializer,
    Microbatches,
    Objective,
    OptimizerConstructor,
    ParameterGroups,
    ParameterObserver,
    StepExecution,
)
from .backends._spec import StepSpec
from .forward import Forward
from .observations import MetricSummary, StepObservations


@dataclass(frozen=True)
class StepResult(StepObservations):
    step: int = 0
    seconds: float = 0.0
    hyperparams: Mapping[str, Any] = field(default_factory=dict)
    summary: MetricSummary = field(default_factory=MetricSummary)

    @property
    def loss(self) -> float:
        return sum(self.losses)


@dataclass(frozen=True)
class EvaluationResult:
    losses: tuple[float, ...]
    metrics: tuple[Any, ...]
    seconds: float
    summary: MetricSummary = field(default_factory=MetricSummary)

    @property
    def mean_loss(self) -> float:
        return sum(self.losses) / len(self.losses)


class Trainer:
    """One source item is one update; objective and microbatching are user code."""

    def __init__(
        self,
        model: nn.Module,
        *,
        objective: Objective,
        optimizer: OptimizerConstructor,
        optimizer_args: Mapping[str, Any] | None = None,
        schedules: Mapping[str, Callable[[int], Any]] | None = None,
        hyperparams: Sequence[str] = (),
        parameter_groups: ParameterGroups | None = None,
        microbatches: Microbatches | Mapping[str, Microbatches] | None = None,
        backend: Backend | None = None,
        master_dtype: torch.dtype | None = None,
        grad_dtype: torch.dtype | None = None,
        parameter_metrics: ParameterObserver | None = None,
        metric_reducer: Callable[
            [StepObservations], MetricSummary | Mapping[str, float]
        ]
        | None = None,
        eval_fn: Objective | None = None,
        distributed: Distributed | None = None,
        shard_optimizer: bool = True,
    ) -> None:
        if backend is None:
            from .backends import PyTorch

            backend = PyTorch()
        self.model = model
        self.objective = objective
        self.eval_fn = eval_fn or objective
        self.backend = backend
        self.distributed = distributed
        self._distributed = None
        self.schedules = dict(schedules or {})
        if any(not callable(fn) for fn in self.schedules.values()):
            raise TypeError("each schedule must be callable(step)")
        self.hyperparams = tuple(dict.fromkeys((*hyperparams, *self.schedules)))
        self._optimizer = functools.partial(
            OptimizerFactory.bind,
            constructor=optimizer,
            arguments=dict(optimizer_args or {}),
            parameter_groups=parameter_groups,
        )
        self._spec = StepSpec(
            ScaledObjective(objective),
            self._optimizer,
            self.hyperparams,
            master_dtype,
            grad_dtype,
            parameter_metrics,
            distributed,
            shard_optimizer,
        )
        self.candidates = candidate_functions(microbatches)
        self.selected_candidate: str | None = None
        self.metric_reducer = metric_reducer
        self.step_count = 0
        self._elapsed_seconds = 0.0
        self._elapsed_started: float | None = None
        self._execution: StepExecution | None = None
        self._evaluation: Forward | None = None
        self._source: Any = None
        self._source_state: Any = None
        self._closed = False

    @property
    def elapsed_seconds(self) -> float:
        if self._elapsed_started is None:
            return self._elapsed_seconds
        return self._elapsed_seconds + time.perf_counter() - self._elapsed_started

    @property
    def plan(self) -> Any:
        return None if self._execution is None else self._execution.plan

    @property
    def planning(self) -> Any:
        return None if self._execution is None else self._execution.planning

    def prepare(
        self,
        example_data: Any,
        *,
        initialize: Initializer | None = None,
        checkpoint: str | Path | None = None,
    ) -> Self:
        if self._closed or self._execution is not None:
            raise RuntimeError("Trainer must be open and unprepared")
        if self.distributed is not None:
            bind = getattr(self.backend, "_distributed_binding", None)
            if bind is None:
                raise NotImplementedError(
                    "this backend does not support distributed ownership"
                )
            self._distributed = bind(
                self.model, self.distributed, shard_optimizer=self._spec.shard_optimizer
            )
        state, loop = (
            (None, None)
            if checkpoint is None
            else _checkpoint.load(checkpoint, distributed=self._distributed)
        )
        omitted: set[str] = set()
        if state is not None and self._distributed is not None:
            from shadowspill.pytorch.distributed._checkpoint import (
                master_aliases,
                restore_compute_weights,
            )

            omitted = master_aliases(state, self._distributed)
        validate_initialization(
            self.model,
            initialize=initialize,
            state=None if state is None else state["model"],
        )

        def fill(model: nn.Module) -> None:
            initialize_model(
                model,
                initialize=initialize,
                state=None if state is None else state["model"],
                missing_parameters=omitted,
            )
            if state is not None and self._distributed is not None:
                restore_compute_weights(
                    dict(model.named_parameters()), state["masters"], self._distributed
                )

        prepare_model = getattr(self.backend, "initialize_model", None)
        if prepare_model is None:
            fill(self.model)
        else:
            self.model = prepare_model(self.model, fill)

        candidates = self.candidates
        if loop is not None:
            selected = loop["selected_candidate"]
            if selected not in candidates:
                raise ValueError(f"checkpoint candidate {selected!r} was not supplied")
            candidates = {selected: candidates[selected]}
        examples = {
            name: make_microbatches(fn, example_data) for name, fn in candidates.items()
        }
        modes = {name: m.training for name, m in self.model.named_modules()}
        with model_mode(self.model, True):
            self._execution, self.selected_candidate = self.backend.prepare_step(
                self.model, self._spec, examples
            )
        self.model = self._execution.model
        for name, module in self.model.named_modules():
            module.training = modes[name]
        if state is not None:
            self._execution.load_state_dict(state)
            assert loop is not None
            self._restore_loop(loop)
        if self._elapsed_started is None:
            self._elapsed_started = time.perf_counter()
        return self

    def _require_prepared(self) -> StepExecution:
        if self._execution is None or self._closed:
            raise RuntimeError("call prepare before using an open Trainer")
        return self._execution

    def step(
        self, data: Any, *, hyperparams: Mapping[str, Any] | None = None
    ) -> StepResult:
        execution = self._require_prepared()
        assert self.selected_candidate is not None
        values = {name: fn(self.step_count) for name, fn in self.schedules.items()}
        values.update(hyperparams or {})
        undeclared = set(values).difference(self.hyperparams)
        if undeclared:
            raise ValueError(
                f"undeclared dynamic hyperparameters: {sorted(undeclared)}"
            )
        batches = make_microbatches(self.candidates[self.selected_candidate], data)
        started = time.perf_counter()
        with model_mode(self.model, True):
            losses, metrics, parameters = execution.run(batches, values)
        execution.synchronize()
        host = StepObservations.collect(losses, metrics, parameters)
        seconds = time.perf_counter() - started
        self.step_count += 1
        summary = self._reduce(host)
        return StepResult(
            host.losses,
            host.metrics,
            host.parameter_metrics,
            self.step_count,
            seconds,
            values,
            summary,
        )

    def diagnose(
        self, data: Any, *, directory: str | Path, warmup: int = 1
    ) -> Mapping[str, Any]:
        """Warm up and trace a zero-LR step without advancing training or logging.

        Requires a backend with step diagnostics and an optimizer advertising
        zero_lr_preserves_state. Model buffers and RNG are preserved separately.
        The supplied data is reused; this method never advances a data source.
        """
        execution = self._require_prepared()
        if not isinstance(warmup, int) or warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        diagnose = getattr(execution, "diagnose", None)
        if not callable(diagnose):
            raise ValueError("this backend does not support startup step timelines")
        assert self.selected_candidate is not None
        rng = _checkpoint.rng_state(self.backend.device)
        started = time.perf_counter()
        try:
            batches = make_microbatches(self.candidates[self.selected_candidate], data)
            with model_mode(self.model, True):
                return cast(
                    Mapping[str, Any], diagnose(batches, Path(directory), warmup=warmup)
                )
        finally:
            _checkpoint.restore_rng(rng, self.backend.device)
            if self._elapsed_started is not None:
                self._elapsed_started += time.perf_counter() - started

    def _reduce(self, observations: StepObservations) -> MetricSummary:
        if self.metric_reducer is None:
            return MetricSummary()
        summary = self.metric_reducer(observations)
        if isinstance(summary, Mapping):
            return MetricSummary(scalars=dict(summary))
        if not isinstance(summary, MetricSummary):
            raise TypeError("metric_reducer must return a mapping or MetricSummary")
        return summary

    def evaluate(
        self,
        source: Iterable[Any] | Callable[[], Iterable[Any]],
        *,
        batches: int | None = None,
    ) -> EvaluationResult:
        self._require_prepared()
        if batches is not None and batches < 1:
            raise ValueError("eval batches must be positive")
        assert self.selected_candidate is not None
        updates = source() if callable(source) else source
        updates = iter(updates) if batches is None else islice(updates, batches)
        losses: list[float] = []
        metrics: list[tuple[Any, ...]] = []
        contributions: list[float] = []
        started = time.perf_counter()
        for data in updates:
            parts = make_microbatches(self.candidates[self.selected_candidate], data)
            if self._evaluation is None:
                objective = ScaledObjective(self.eval_fn)
                self._evaluation = Forward(
                    self.model,
                    forward_fn=lambda model, pair, objective=objective: objective(
                        model, *pair
                    ),
                    backend=self.backend,
                    distributed=self.distributed,
                ).prepare(parts[0])
            total = 0.0
            observations: list[Any] = []
            for part in parts:
                loss, values = self._evaluation(part)
                self._evaluation.synchronize()
                host = StepObservations.collect((loss,), (values,))
                total += host.losses[0]
                contributions.extend(host.losses)
                observations.extend(host.metrics)
            losses.append(total)
            metrics.append(tuple(observations))
        if not losses:
            raise ValueError("evaluation source produced no updates")
        return EvaluationResult(
            tuple(losses),
            tuple(metrics),
            time.perf_counter() - started,
            self._reduce(
                StepObservations(
                    tuple(contributions),
                    tuple(metric for update in metrics for metric in update),
                )
            ),
        )

    def fit(
        self,
        source: Iterable[Any],
        *,
        steps: int,
        run_dir: str | Path | None = None,
        log_every: int = 1,
        tables_every: int = 100,
        logger: Callable[[dict[str, Any]], None] | None = None,
        callbacks: Sequence[Callable[[Trainer, StepResult], None]] = (),
        eval_data: Iterable[Any] | Callable[[], Iterable[Any]] | None = None,
        eval_every: int = 0,
        eval_batches: int | None = None,
        checkpoint_every: int = 0,
        checkpoint_dir: str | Path | None = None,
        checkpoint_weights: Literal["master", "compute"] = "master",
        keep_last: int = 3,
        startup_diagnostics: bool = False,
    ) -> StepResult | None:
        self._require_prepared()
        if checkpoint_weights not in {"master", "compute"}:
            raise ValueError("checkpoint_weights must be 'master' or 'compute'")
        if not isinstance(steps, int) or steps < self.step_count:
            raise ValueError("steps is the target total completed update count")
        if (
            min(log_every, tables_every, eval_every, checkpoint_every) < 0
            or keep_last < 1
        ):
            raise ValueError("cadences must be nonnegative and keep_last positive")
        if eval_every and eval_data is None:
            raise ValueError("eval_every requires eval_data")
        if checkpoint_every and checkpoint_dir is None and run_dir is None:
            raise ValueError("checkpointing needs checkpoint_dir or run_dir")
        if startup_diagnostics and run_dir is None:
            raise ValueError("startup_diagnostics requires run_dir")
        self._source = source
        if self._source_state is not None:
            restore = getattr(source, "load_state_dict", None)
            if not callable(restore):
                raise ValueError(
                    "checkpoint has source progress; supply a resumable source"
                )
            restore(self._source_state)
            self._source_state = None
        iterator = iter(source)
        run_root = None if run_dir is None else Path(run_dir)
        root = run_root
        if root is not None and self._distributed is not None:
            root = root / f"rank-{self._distributed.control.rank:05d}"
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            if self.planning is not None:
                self.planning.save(root / "search.json")
            if self.plan is not None:
                self._require_prepared().save_plan(root / "plan.json")
        if startup_diagnostics and self.step_count < steps:
            assert root is not None
            try:
                first = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "data source ended before startup diagnostics"
                ) from error
            self.diagnose(first, directory=root / "startup")
            iterator = chain((first,), iterator)
        log = None if root is None else (root / "metrics.jsonl").open("a", buffering=1)
        last = None
        try:
            while self.step_count < steps:
                try:
                    data = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        f"data source ended after {self.step_count} of {steps} updates"
                    ) from error
                last = self.step(data)
                if _due(log_every, self.step_count, steps):
                    record = {
                        "step": last.step,
                        "train/loss": last.loss,
                        "train/step_seconds": last.seconds,
                        "train/elapsed_seconds": self.elapsed_seconds,
                        **{
                            "hyperparameters/" + k: v
                            for k, v in last.hyperparams.items()
                        },
                        **{"train/" + k: v for k, v in last.summary.scalars.items()},
                    }
                    _log(record, log, logger)
                    if _due(tables_every, self.step_count, steps):
                        _log_tables(last.summary, "train", last.step, root, logger)
                if _due(eval_every, self.step_count, steps):
                    assert eval_data is not None
                    evaluated = self.evaluate(eval_data, batches=eval_batches)
                    _log(
                        {
                            "step": self.step_count,
                            "eval/loss": evaluated.mean_loss,
                            "eval/seconds": evaluated.seconds,
                            **{
                                "eval/" + k: v
                                for k, v in evaluated.summary.scalars.items()
                            },
                        },
                        log,
                        logger,
                    )
                    _log_tables(
                        evaluated.summary, "eval", self.step_count, root, logger
                    )
                if _due(checkpoint_every, self.step_count, steps):
                    if checkpoint_dir is None:
                        assert run_root is not None
                        directory = run_root / "checkpoints"
                    else:
                        directory = Path(checkpoint_dir)
                    self.save(
                        directory / f"step_{self.step_count:08d}",
                        weights=checkpoint_weights,
                    )
                    if (
                        self._distributed is None
                        or self._distributed.control.rank
                        == self._distributed.control.members[0]
                    ):
                        _prune(directory, keep_last)
                for callback in callbacks:
                    callback(self, last)
        finally:
            if log is not None:
                log.close()
        return last

    def save(
        self,
        path: str | Path,
        *,
        source: Any = None,
        weights: Literal["master", "compute"] = "master",
    ) -> Path:
        """Save masters by default, or compute weights with ``weights="compute"``.

        Only one representation is stored. A compute checkpoint restores masters
        by upcasting and therefore cannot recover their original extra precision.
        """
        execution = self._require_prepared()
        source = self._source if source is None else source
        state_dict = getattr(source, "state_dict", None)
        data_state = state_dict() if callable(state_dict) else None
        schedules = {}
        for name, schedule in self.schedules.items():
            state_dict = getattr(schedule, "state_dict", None)
            if callable(state_dict):
                schedules[name] = state_dict()
        return _checkpoint.save(
            path,
            execution,
            {
                "step": self.step_count,
                "elapsed_seconds": self.elapsed_seconds,
                "selected_candidate": self.selected_candidate,
                "schedules": schedules,
                "source": data_state,
                "rng": _checkpoint.rng_state(self.backend.device),
            },
            distributed=self._distributed,
            weights=weights,
        )

    def load(self, path: str | Path) -> Self:
        """Restore a prepared runner, preserving its initialized caches."""
        execution = self._require_prepared()
        state, loop = _checkpoint.load(path, distributed=self._distributed)
        execution.synchronize()
        execution.load_state_dict(state)
        self._restore_loop(loop)
        return self

    def _restore_loop(self, state: Mapping[str, Any]) -> None:
        if state["selected_candidate"] != self.selected_candidate:
            raise ValueError(
                "checkpoint microbatch candidate differs from the prepared plan"
            )
        self.step_count = state["step"]
        self._elapsed_seconds = state["elapsed_seconds"]
        self._elapsed_started = time.perf_counter()
        for name, value in state["schedules"].items():
            restore = getattr(self.schedules.get(name), "load_state_dict", None)
            if not callable(restore):
                raise ValueError(f"schedule {name!r} cannot restore saved state")
            restore(value)
        self._source_state = state["source"]
        _checkpoint.restore_rng(state["rng"], self.backend.device)

    def close(self) -> None:
        if self._closed:
            return
        if self._evaluation is not None:
            self._evaluation.close()
        if self._execution is not None:
            self._execution.close()
        self._elapsed_seconds = self.elapsed_seconds
        self._elapsed_started = None
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def _due(every: int, done: int, target: int) -> bool:
    return every > 0 and (done % every == 0 or done == target)


def _log(
    record: dict[str, Any],
    log: TextIO | None,
    logger: Callable[[dict[str, Any]], None] | None,
) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if log is not None:
        log.write(line + "\n")
    if logger is not None:
        logger(record)


def _prune(directory: Path, keep_last: int) -> None:
    checkpoints = sorted(
        path
        for path in directory.glob("step_*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path)


def _log_tables(
    summary: MetricSummary,
    phase: str,
    step: int,
    root: Path | None,
    logger: Callable[[dict[str, Any]], None] | None,
) -> None:
    for name, table in summary.tables.items():
        key = phase + "/" + name
        if root is not None:
            with (root / "observations.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "step": step,
                            "name": key,
                            "columns": table.columns,
                            "rows": table.rows,
                        }
                    )
                    + "\n"
                )
        emit = getattr(logger, "table", None)
        if callable(emit):
            emit(step, key, table)
