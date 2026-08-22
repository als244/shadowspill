"""Mutable timing state used only while one execution trace is armed."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from shadowspill.ir import MemoryAction
from shadowspill.pytorch.lowering.training import TrainingTaskEntrypoint
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.simulator import SimulationResult


@dataclass(slots=True)
class ArmedTaskTiming:
    """Reusable event and host-clock state for one execution task."""

    entrypoint: TrainingTaskEntrypoint
    expected_profile_seconds: float
    execution_ordinal: int
    semantic_name: str
    readiness_event: torch.cuda.Event
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    dispatch_started_ns: int = 0
    dispatch_finished_ns: int = 0
    dispatch_before_finished_ns: int = 0
    dispatch_after_started_ns: int = 0
    dispatch_stream_resolution_ns: int = 0
    dispatch_readiness_marker_ns: int = 0
    dispatch_runtime_before_task_ns: int = 0
    dispatch_input_lookup_ns: int = 0
    dispatch_storage_rebind_ns: int = 0
    dispatch_argument_assembly_ns: int = 0
    dispatch_rebind_ns: int = 0
    dispatch_invoke_ns: int = 0
    dispatch_output_flatten_ns: int = 0
    dispatch_output_classification_ns: int = 0
    dispatch_output_adoption_ns: int = 0
    dispatch_output_state_publish_ns: int = 0
    dispatch_gradient_accumulation_ns: int = 0
    dispatch_output_publish_ns: int = 0
    dispatch_dematerialize_ns: int = 0
    dispatch_postprocess_ns: int = 0
    dispatch_runtime_after_task_ns: int = 0
    dispatch_cleanup_ns: int = 0


@dataclass(slots=True)
class ArmedExecutionTiming:
    """Mutable trace state spanning one complete planned invocation."""

    origin_event: torch.cuda.Event
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    tasks: dict[str, ArmedTaskTiming]
    task_order: tuple[str, ...]
    started: bool = False
    finished: bool = False
    dispatch_call_started_ns: int = 0
    dispatch_call_finished_ns: int = 0
    dispatch_startup_wait_ns: int = 0
    dispatch_initial_actions_ns: int = 0
    stream: torch.cuda.Stream | None = None
    statistics_before: AdapterStatistics | None = None
    actions: tuple[MemoryAction, ...] = ()
    simulation: SimulationResult | None = None
    trace_setup_ns: int = 0


@dataclass(slots=True)
class ArmedSpanTiming:
    """Two-event production-like selected-task timing bracket."""

    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    started: bool = False
    finished: bool = False


__all__ = ["ArmedExecutionTiming", "ArmedSpanTiming", "ArmedTaskTiming"]
