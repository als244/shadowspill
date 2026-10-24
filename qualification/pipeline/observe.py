"""Reusable gate timing/events and optional native traces around Executor.run."""

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch

from qualification.runtime_evidence import adapter_statistics, statistics_dict
from shadowspill import Executor
from shadowspill.runtime.timing import Marker
from shadowspill.runtime.trace import (
    begin_runtime_trace,
    end_runtime_trace,
    prepare_runtime_trace,
    read_runtime_trace,
)


class Observe:
    def __init__(self, executor: Executor, *, steps: int):
        self.executor = executor
        self.handle = executor.runtime._runtime_handle
        self.events = [
            torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            for _ in range(steps + 1)
        ]
        for event in self.events:
            event.record()
        torch.cuda.synchronize()
        self.origin = Marker(self.handle)
        plan = executor.plan
        assert plan.memory_layout is not None
        # Some tasks (notably unfused recurrent attention) allocate thousands of
        # intermediates. Task count alone does not bound their retirement records.
        allocations = len(plan.memory_layout.task_allocation_leases.task)
        capacity = max(
            4096,
            32 * (len(plan.program.tasks) + len(plan.schedule.actions))
            + 8 * allocations,
        )
        prepare_runtime_trace(
            self.handle, event_capacity=capacity, allocation_event_capacity=capacity
        )

    def idle(self) -> None:
        self.executor.synchronize()
        for path in self.executor._paths:
            path.bridge.wait_runtime_idle()

    def traced(
        self,
        args: Sequence[tuple[Any, ...]],
        *,
        step: int,
        hyperparams: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        self.idle()
        before = statistics_dict(adapter_statistics())
        self.origin.record(int(torch.cuda.current_stream().cuda_stream))
        begin_runtime_trace(self.handle, step_id=step, origin=self.origin)
        started = time.perf_counter()
        try:
            self.events[0].record()
            result = self.executor.run(args=args, hyperparams=hyperparams).result()
            self.idle()
            self.events[1].record()
            self.events[1].synchronize()
        finally:
            end_runtime_trace(self.handle)
        elapsed = time.perf_counter() - started
        trace = read_runtime_trace(self.handle)
        if trace.event_overflow or trace.allocation_event_overflow:
            del result
            raise AssertionError(
                "qualification runtime trace overflowed: "
                f"events={trace.event_overflow}, "
                f"allocations={trace.allocation_event_overflow}"
            )
        after = statistics_dict(adapter_statistics())
        summary = {
            "gpu_seconds": self.events[0].elapsed_time(self.events[1]) / 1000,
            "host_seconds": elapsed,
            **{
                key: after["runtime"][key] - before["runtime"][key]
                for key in (
                    "bytes_fetched",
                    "bytes_evicted",
                    "fetch_transfers",
                    "evict_transfers",
                )
            },
        }
        return result, summary, asdict(trace)

    def close(self) -> None:
        self.idle()
        self.origin.release()
