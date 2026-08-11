"""Documented forward/training result values and planned callable types."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from shadowspill.ir import ExecutionPlan, MemoryAction, TaskProfile

from .executor import ForwardExecutor
from .guards import InputSignature
from .materialization import MaterializedForwardState


@dataclass(frozen=True, slots=True)
class PlanReport:
    """Immutable planning, profiling, schedule, and physical-admission evidence."""

    mode: str
    capture_identity: str
    execution_plan: ExecutionPlan
    task_profiles: tuple[TaskProfile, ...]
    transfer_actions: tuple[MemoryAction, ...]
    transfer_bytes_to_host: int
    transfer_bytes_to_device: int
    profile_unique_keys: int
    profile_cache_hits: int
    profile_cache_misses: int
    profiling_provenance: tuple[str, ...]
    phase_timings_ns: tuple[tuple[str, int], ...]

    @property
    def predicted_device_peak_bytes(self) -> int:
        return self.execution_plan.prediction.device_peak_bytes

    @property
    def predicted_host_peak_bytes(self) -> int:
        return self.execution_plan.prediction.host_peak_bytes

    @property
    def predicted_makespan_ns(self) -> int:
        return self.execution_plan.prediction.makespan_ns


class PlannedForward:
    """Forward-only callable returned by :func:`forward_pass`.

    The original model is runtime-owned until `close()`. Calls validate the
    complete fixed input signature before writing an input slot or launching a
    task. Returned tensors are ordinary caller-owned allocator records.
    """

    def __init__(
        self,
        model: nn.Module,
        signature: InputSignature,
        executor: ForwardExecutor,
        state: MaterializedForwardState,
        report: PlanReport,
    ) -> None:
        self._model = model
        self._signature = signature
        self._executor = executor
        self._state = state
        self.plan_report = report
        self._closed = False

    def __call__(self, inputs: Sequence[Any]) -> object:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        self._signature.validate(inputs)
        return self._executor(inputs)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Synchronously snapshot model state into ordinary CPU tensors."""

        if self._closed:
            return OrderedDict(self._model.state_dict())
        return self._state.state_dict()

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Synchronously replace the current host-authoritative model state."""

        if self._closed:
            self._model.load_state_dict(state)
            return
        self._state.load_state_dict(state)

    def close(self) -> None:
        """Synchronize, restore the original model to CPU, and release the plan."""

        if self._closed:
            return
        self._state.restore_cpu_and_unregister()
        self._closed = True

    def __enter__(self) -> PlannedForward:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()


__all__ = ["PlanReport", "PlannedForward"]


def forward_pass(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    device_budget: int,
    host_budget: int,
    partition: str = "auto",
) -> PlannedForward:
    """Plan one fixed-shape forward program around ordinary PyTorch tasks.

    Planning installs ShadowSpill's process-global CUDA allocator. The original
    model remains runtime-owned until the returned callable is closed.
    """

    from .session import build_forward

    return build_forward(
        model,
        example_inputs=example_inputs,
        device_budget=device_budget,
        host_budget=host_budget,
        partition=partition,
    )


__all__ = ["PlanReport", "PlannedForward", "forward_pass"]
