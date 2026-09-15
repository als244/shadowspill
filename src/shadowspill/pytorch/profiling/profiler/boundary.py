"""The allocator boundary a profiled task runs behind: scopes, drains, readings."""

from __future__ import annotations

import ctypes
import itertools
import statistics
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.accelerator import accelerator_device
from shadowspill.pytorch.runtime_adapter.abi import (
    PROFILING_SCOPE_BASE,
    AdapterStatistics,
    Allocation,
)
from shadowspill.pytorch.runtime_adapter.failures import wait_allocator_idle
from shadowspill.pytorch.runtime_adapter.telemetry import AllocationTelemetryError

#: Profiling scope ids, minted once per process rather than once per profiler.
#: A profiler is built per planning call, so a per-instance counter restarted at
#: the base every call and two scopes in different calls wore the same number --
#: which made an allocation's origin ambiguous exactly when several planning
#: calls had run, and that is when it matters.
profiling_scope_ids = itertools.count(PROFILING_SCOPE_BASE)


class AllocatorBoundary:
    """One plan's profiling scopes on the installed allocator.

    Every invocation of a profiled task runs inside an allocation scope named
    after the plan, so the runtime can attribute what the task allocates, and
    the stream is drained before the next one, so nothing outstanding from a
    previous invocation is charged to it.
    """

    def __init__(
        self,
        library: Any,
        *,
        runtime_handle: int,
        plan_id: int,
        device_ordinal: int,
        telemetry_capacity: int,
    ) -> None:
        self.library = library
        self.runtime_handle = runtime_handle
        # Named on every allocation scope: a scope runs outside any task,
        # so the runtime cannot infer which plan the probe is measuring for.
        self.plan_id = plan_id
        self.device_ordinal = device_ordinal
        self.telemetry_capacity = telemetry_capacity
        self.conditioned = False
        self._timing_events: tuple[Any, Any] | None = None

    def stream(self) -> torch.cuda.Stream:
        """Select the profiled device and return its current stream."""

        torch.cuda.set_device(self.device_ordinal)
        return torch.cuda.current_stream(self.device_ordinal)

    @contextmanager
    def scope(self, stream: torch.cuda.Stream) -> Iterator[int]:
        """Open one allocation scope around the body; abort it if the body raises."""

        scope_id = next(profiling_scope_ids)
        status = int(
            self.library.shadowspill_pytorch_allocation_scope_begin(
                self.plan_id, scope_id
            )
        )
        if status != 0:
            raise CaptureError(
                f"profiling allocation scope begin failed with status {status}"
            )
        try:
            yield scope_id
        except BaseException:
            self.library.shadowspill_pytorch_allocation_scope_abort()
            raise
        status = int(
            self.library.shadowspill_pytorch_allocation_scope_end(
                scope_id, stream.cuda_stream
            )
        )
        if status != 0:
            raise CaptureError(
                f"profiling allocation scope end failed with status {status}"
            )

    def invoke(self, task: Callable[[], object], stream: torch.cuda.Stream) -> None:
        """Run the task once inside a scope and drain before returning."""

        with self.scope(stream):
            output = task()
            del output
        # Drain before the next invocation. Retirement is asynchronous, so
        # without this a warmup loop runs every iteration while the previous
        # ones still hold their ranges, leaving more outstanding than a pool
        # sized to fit the task being measured can hold.
        self.drain(stream, problem="task warmup")

    def time_once(self, task: Callable[[], object], stream: torch.cuda.Stream) -> int:
        """Run the task once between timing events; return its duration in ns."""

        start, finish = self._events()
        with self.scope(stream):
            start.record(stream)
            output = task()
            del output
            finish.record(stream)
        finish.synchronize()
        self.require_idle(problem="task timing sample")
        elapsed_ms = cast(float, start.elapsed_time(finish))
        return max(0, round(elapsed_ms * 1_000_000))

    def condition_device(self, stream: torch.cuda.Stream) -> None:
        """Warm clocks and provider state once using bounded preallocated GEMM."""

        shape = (2048, 2048)
        device = accelerator_device(self.device_ordinal)
        left = torch.randn(shape, dtype=torch.bfloat16, device=device)
        right = torch.randn(shape, dtype=torch.bfloat16, device=device)
        output = torch.empty(shape, dtype=torch.bfloat16, device=device)
        samples: list[int] = []
        start, finish = self._events()
        for _ in range(64):
            start.record(stream)
            torch.mm(left, right, out=output)
            finish.record(stream)
            finish.synchronize()
            samples.append(max(1, round(start.elapsed_time(finish) * 1_000_000)))
            if len(samples) >= 3:
                recent = samples[-3:]
                median = float(statistics.median(recent))
                if median > 0 and (max(recent) - min(recent)) / median <= 0.02:
                    break
        del output
        del right
        del left
        self.drain(stream, problem="device conditioning")
        self.conditioned = True

    def drain(self, stream: torch.cuda.Stream, *, problem: str) -> None:
        """Wait for the stream, then for the allocator to retire what it freed."""

        stream.synchronize()
        self.require_idle(problem=problem)

    def require_idle(self, *, problem: str) -> None:
        """Block on the runtime's progress-safe quiescence boundary."""

        message = wait_allocator_idle(
            self.library, self.runtime_handle, problem=problem
        )
        if message is not None:
            raise AllocationTelemetryError(message)

    def requested_allocated_bytes(self) -> int:
        """Bytes the process holds in the pool right now, as requested."""

        pool = self.statistics().allocator_pool
        return int(pool.requested_allocated_bytes)

    def events_overflowed(self) -> bool:
        """Whether the last measurement filled the allocation event record."""

        return bool(self.statistics().runtime.allocation_event_overflow)

    def statistics(self) -> AdapterStatistics:
        statistics = AdapterStatistics()
        status = int(
            self.library.shadowspill_pytorch_allocator_statistics(
                ctypes.byref(statistics)
            )
        )
        if status != 0:
            raise AllocationTelemetryError(
                f"allocator statistics failed during profiling with status {status}"
            )
        return statistics

    def allocation_for_pointer(self, address: int) -> Allocation:
        """The slab allocation holding an address; an error for any other memory."""

        allocation = Allocation()
        status = int(
            self.library.shadowspill_pytorch_allocation_for_pointer(
                address, ctypes.byref(allocation)
            )
        )
        if status != 0:
            raise CaptureError(
                "compiled task returned storage outside the ShadowSpill slab"
            )
        return allocation

    def _events(self) -> tuple[Any, Any]:
        if self._timing_events is None:
            event_factory: Any = torch.cuda.Event
            self._timing_events = (
                event_factory(enable_timing=True),
                event_factory(enable_timing=True),
            )
        return self._timing_events
