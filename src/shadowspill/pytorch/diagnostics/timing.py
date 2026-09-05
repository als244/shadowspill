"""Mutable timing state used only while one execution trace is armed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from shadowspill.ir import MemoryAction
from shadowspill.planner.diagnostics.mapping import FrozenMapping
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
    inputs_ready_event: torch.cuda.Event
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    #: Four instants that partition one task's frontend cycle with no gap:
    #: entering the opening boundary, leaving it for the compiled call, the
    #: call returning, and leaving the closing boundary. Every duration
    #: between boundaries is a difference of two of these, so none is stored.
    before_task_enter_ns: int = 0
    before_task_exit_ns: int = 0
    after_task_enter_ns: int = 0
    after_task_exit_ns: int = 0
    dispatch_input_lookup_ns: int = 0
    dispatch_storage_rebind_ns: int = 0
    dispatch_input_acquire_ns: int = 0
    dispatch_allocation_reuse_ns: int = 0
    dispatch_argument_assembly_ns: int = 0
    dispatch_output_flatten_ns: int = 0
    dispatch_output_classification_ns: int = 0
    dispatch_output_adoption_ns: int = 0
    dispatch_output_state_publish_ns: int = 0
    dispatch_output_publish_ns: int = 0
    dispatch_dematerialize_ns: int = 0
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
    prior_invocation_drain_ns: int = 0
    dispatch_initial_actions_ns: int = 0
    stream: torch.cuda.Stream | None = None
    statistics_before: AdapterStatistics | None = None
    actions: tuple[MemoryAction, ...] = ()
    simulation: SimulationResult | None = None
    trace_setup_ns: int = 0
    #: Every read and write of each alias group by the selected tasks, as
    #: (execution ordinal, is_write) in execution order: what a transfer's
    #: record uses to name the task whose result it carries and the task
    #: that will read it.
    alias_accesses: Mapping[str, tuple[tuple[int, bool], ...]] = field(
        default_factory=lambda: FrozenMapping({})
    )


@dataclass(slots=True)
class ArmedSpanTiming:
    """Two-event production-like selected-task timing bracket."""

    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    started: bool = False
    finished: bool = False


__all__ = ["ArmedExecutionTiming", "ArmedSpanTiming", "ArmedTaskTiming"]
