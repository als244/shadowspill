"""Admitted saved Plans, one worker, one outstanding asynchronous invocation."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Self

import torch

from shadowspill.artifacts.plan import Plan
from shadowspill.communication.binding.ranks import RankAssignment
from shadowspill.pytorch.invocation import InvocationResult, ReusableCompletionEvent
from shadowspill.runtime.plan.annotations import TaskBoundaryAnnotations

from .dispatch import dispatch
from .distributed.checks import check_external_budget, check_invocation, coordinated
from .inputs import stage_inputs
from .paths import ExecutionPath
from .results import TrainingResult, collect_outputs
from .tensors import TensorChunk, visit_tensors
from .tracing import InvocationTrace
from .worker import create_paths


class Executor:
    def __init__(
        self,
        plan: Plan,
        *,
        plans: Mapping[str, Plan] | None = None,
        device: int = 0,
        rank_map: Mapping[int, int] | None = None,
        external_headroom_bytes: int | None = None,
        parameter_norms: bool = False,
    ) -> None:
        self.parameter_norms = parameter_norms
        world_size = (
            1
            if plan.program.parallelism is None
            else plan.program.parallelism.world_size
        )
        # Distributed assignment is validated collectively during startup, so
        # one invalid map cannot leave peers entering NCCL alone.
        self._assignment = (
            RankAssignment.resolve(1, rank_map) if world_size == 1 else None
        )
        self.plan = plan
        self.plans = MappingProxyType(dict(plans or {}))
        self.device = torch.device("cuda", device)
        self._pending: InvocationResult[Any] | None = None
        self._closed = False
        self._plan_released = False
        self._initialized = False
        self._failure: BaseException | None = None
        (
            self.runtime,
            self._paths,
            self._named_paths,
            self._streams,
            self._distributed,
            self._observations,
            self._providers,
        ) = create_paths(
            plan, device, external_headroom_bytes, self.plans, rank_map, parameter_norms
        )
        if self._distributed is not None:
            self._assignment = self._distributed.groups.assignment
        self._completion = ReusableCompletionEvent(
            self.runtime._runtime_handle, torch.device("cuda", device)
        )

    @property
    def logical_rank(self) -> int:
        """This worker's identity in the saved Plan, independent of launcher rank."""
        return 0 if self._distributed is None else self._distributed.groups.logical_rank

    def initialize(
        self,
        fill_tensor: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
        restore: bool = False,
    ) -> None:
        self.synchronize()
        self._initialized = False

        def fill() -> None:
            path = self._paths[0]
            visit_tensors(
                path.storage, fill_tensor, chunk_bytes, initialize=True, restore=restore
            )
            path.controls.initialize(
                fill_tensor, chunk_bytes=chunk_bytes, restore=restore
            )
            for name, extra in self._named_paths.items():
                visit_tensors(
                    extra.storage,
                    fill_tensor,
                    chunk_bytes,
                    initialize=True,
                    restore=restore,
                    skip_shared=True,
                    prefix=f"plans.{name}.",
                )

        coordinated(self._distributed, fill)
        self._initialized = True

    def export_tensors(
        self,
        write_chunk: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int = 64 << 20,
    ) -> None:
        self.synchronize()
        if not self._initialized:
            raise RuntimeError("initialize persistent tensors before exporting them")
        visit_tensors(
            self._paths[0].storage, write_chunk, chunk_bytes, initialize=False
        )
        self._paths[0].controls.export(write_chunk, chunk_bytes=chunk_bytes)
        for name, path in self._named_paths.items():
            visit_tensors(
                path.storage,
                write_chunk,
                chunk_bytes,
                initialize=False,
                skip_shared=True,
                prefix=f"plans.{name}.",
            )

    def run(
        self,
        *,
        plan: str | None = None,
        args: Sequence[tuple[Any, ...]] | None = None,
        kwargs: Sequence[Mapping[str, Any]] | None = None,
        input_loader: Callable[[int], tuple[tuple[Any, ...], Mapping[str, Any]]]
        | None = None,
        hyperparams: Mapping[str, Any] | None = None,
        accumulate_gradients: bool = False,
        collect_parameter_norms: bool | None = None,
        profiler_annotations: bool = False,
        runtime_trace: bool = False,
    ) -> InvocationResult[Any]:
        self.synchronize()
        observe = (
            self.parameter_norms and plan is None
            if collect_parameter_norms is None
            else collect_parameter_norms
        )
        if self._distributed is not None:
            check_invocation(self._distributed, plan, accumulate_gradients, observe)

        def prepare() -> ExecutionPath:
            if type(profiler_annotations) is not bool:
                raise TypeError("profiler_annotations must be a bool")
            if type(runtime_trace) is not bool:
                raise TypeError("runtime_trace must be a bool")
            if type(observe) is not bool:
                raise TypeError("collect_parameter_norms must be a bool or None")
            if observe and (self._observations is None or plan is not None):
                raise ValueError(
                    "enable parameter_norms on Executor and select its training Plan"
                )
            if not self._initialized:
                raise RuntimeError("initialize persistent tensors before run")
            if plan is not None and plan not in self._named_paths:
                raise ValueError(f"unknown admitted Plan: {plan}")
            if accumulate_gradients and (plan is not None or len(self._paths) != 2):
                raise ValueError(
                    "this Plan has no gradient-accumulation invocation path"
                )
            path = (
                self._paths[int(accumulate_gradients)]
                if plan is None
                else self._named_paths[plan]
            )
            path.bridge.require_initial_residency(path.plan.schedule.initial_residency)
            path.controls.apply(hyperparams or {})
            stage_inputs(path.storage, args, kwargs, input_loader)
            return path

        path = coordinated(self._distributed, prepare)
        observations = self._observations if observe else None
        annotations = (
            TaskBoundaryAnnotations(path.bridge) if profiler_annotations else None
        )
        trace = InvocationTrace(path) if runtime_trace else None
        try:
            if trace is not None:
                trace.start()
            if annotations is not None:
                annotations.set_enabled(True)
            parameter_metrics = {} if observations is None else observations.begin()
            with self._streams.invocation() as streams:
                for task in path.tasks:
                    with torch.cuda.stream(streams[task.task.resource]):
                        dispatch(
                            task,
                            path.storage,
                            path.controls,
                            observations,
                            annotations,
                            trace,
                        )
            payload = collect_outputs(path.storage, path.acquisition)
            if observations is not None:
                assert isinstance(payload, TrainingResult)
                payload = replace(payload, parameter_metrics=parameter_metrics)
            wait = self._completion.record()
            if trace is not None:
                trace.dispatched()

            def finish() -> None:
                wait()
                path.bridge.wait_until_idle()
                if trace is not None:
                    trace.finish()
                if annotations is not None:
                    annotations.set_enabled(False)
                if observations is not None:
                    observations.finish(parameter_metrics)
                if self._distributed is not None:
                    coordinated(
                        self._distributed, lambda: check_external_budget(self.runtime)
                    )

            self._pending = InvocationResult(
                payload,
                finish,
                on_resolved=lambda result: None,
                on_failure=self._failed,
                diagnostics=None if trace is None else lambda: trace.result,
            )
            return self._pending
        except BaseException as error:
            if trace is not None:
                try:
                    trace.cancel()
                except BaseException as cleanup:
                    error.add_note(f"Trace cleanup also failed: {cleanup}")
            self._failed(error)
            raise

    def _failed(self, error: BaseException) -> None:
        self._failure = error

    def synchronize(self) -> None:
        if self._closed or self._plan_released:
            raise RuntimeError("Executor is closed")
        if self._failure is not None:
            raise RuntimeError(
                "Executor is unusable after an execution failure"
            ) from self._failure
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def close(self) -> None:
        if self._closed:
            return
        if not self._plan_released:
            try:
                self.synchronize()
            finally:
                self._plan_released = True
                with ExitStack() as cleanup:
                    if self._distributed is not None:
                        cleanup.callback(self._distributed.close)
                    cleanup.callback(self._providers.close)
                    for path in (*self._paths, *self._named_paths.values()):
                        cleanup.callback(path.close, self.runtime)
                    if self._observations is not None:
                        cleanup.callback(self._observations.close)
                    cleanup.callback(self._completion.release)
        # A caller-held output may prevent runtime shutdown. Allow close() to
        # be retried after it is released without destroying plan handles twice.
        self.runtime.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Executor cleanup also failed: {cleanup}")
