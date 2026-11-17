"""Measured saved-Plan invocations; scheduling and state belong to Executor."""

import time
from collections.abc import Mapping
from contextlib import nullcontext

from shadowspill.execution import TrainingResult
from shadowspill.training.observations import MetricSummary, StepObservations, to_host
from shadowspill.training.results import StepResult


def updates(
    executor,
    source,
    *,
    steps,
    schedules,
    metric_reducer,
    parameter_norms_every,
    nsys,
    rank,
):
    schedules = dict(schedules or {})
    names = {
        binding["name"]
        for binding in executor.plan.program.bindings.get("optimizer_bindings", ())
        if binding["role"] == "hyperparameter"
    }
    if schedules.keys() - names or any(
        not callable(value) for value in schedules.values()
    ):
        raise ValueError("schedules must map captured hyperparameters to callables")
    window = None
    if nsys is not None:
        from shadowspill.orchestration.capture._nsys_runtime import UpdateCapture

        window = UpdateCapture(nsys)
    iterator = iter(source)
    for step in range(1, steps + 1):
        try:
            data = dict(next(iterator))
        except StopIteration as error:
            raise RuntimeError(
                f"data source ended after {step - 1} of {steps} updates"
            ) from error
        unknown = data.keys() - {
            "args",
            "kwargs",
            "input_loader",
            "hyperparams",
            "runtime_trace",
            "profiler_annotations",
        }
        if unknown:
            raise ValueError(f"unknown training invocation inputs: {sorted(unknown)}")
        controls = {name: function(step - 1) for name, function in schedules.items()}
        controls.update(data.pop("hyperparams", None) or {})
        captured = window is not None and window.begin(
            step, can_start=True, request_id=f"step-{step}"
        )
        if captured:
            data["profiler_annotations"] = True
        try:
            with (
                nullcontext()
                if window is None
                else window.annotation(f"train/step={step}/rank={rank}")
            ):
                started = time.perf_counter()
                pending = executor.run(
                    **data,
                    hyperparams=controls,
                    collect_parameter_norms=bool(
                        parameter_norms_every and step % parameter_norms_every == 0
                    ),
                )
                result = pending.result()
                try:
                    if not isinstance(result, TrainingResult):
                        raise TypeError("training Plan returned an invalid result")
                    host = StepObservations.collect(
                        result.losses, result.metrics, result.parameter_metrics
                    )
                    diagnostics = (
                        pending.diagnostics if data.get("runtime_trace") else None
                    )
                finally:
                    executor.synchronize()
                    del result, pending
                seconds = time.perf_counter() - started
        except BaseException:
            if window is not None:
                window.failed("Training execution failed during capture.")
            raise
        if window is not None:
            window.complete(step, request_id=f"step-{step}")
        summary = MetricSummary()
        if metric_reducer is not None:
            reduced = metric_reducer(host)
            summary = (
                MetricSummary(scalars=dict(reduced))
                if isinstance(reduced, Mapping)
                else reduced
            )
            if not isinstance(summary, MetricSummary):
                raise TypeError("metric_reducer must return a mapping or MetricSummary")
        yield StepResult(
            host.losses,
            host.metrics,
            host.parameter_metrics,
            step,
            seconds,
            to_host(controls),
            summary,
            diagnostics,
        )
