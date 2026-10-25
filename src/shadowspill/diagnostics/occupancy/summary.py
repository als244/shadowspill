"""Summary for pool-occupancy diagnostics."""

from __future__ import annotations

from .clock import _busy_ns
from .intervals import GIB
from .plan import PlanOccupancy


def summarize(
    result: PlanOccupancy, *, tokens_per_step: int | None = None
) -> dict[str, float | None]:
    """The step in a few numbers, from the same spans the pages draw.

    Seconds are the step's span on this view's clock; idle is the share of
    it with nothing computing; recompute is the plan's regeneration overhead
    -- what its recompute alternatives cost beyond their save alternatives,
    by the program's profiles, the number the planner reports -- as a share
    of the span (unknown without save profiles). A lane's utilisation is the
    share of the span it is busy. Its effective rate counts full execution
    object bytes over complete transfer time, including collective phases.
    Host-copy bytes are reported separately; neither is fabric bandwidth.
    """

    tasks = result.tasks
    end_ns = max(
        [result.clock.makespan_ns]
        + [span.end_ns for span in tasks]
        + [span.end_ns for span in result.transfers]
    )
    # Both clocks start at the step origin, before opening transfers.
    span_ns = max(end_ns, 1)
    compute_ns = sum(span.end_ns - span.start_ns for span in tasks)
    recompute_ns = result.facts.recompute_overhead_ns
    lane_ns = {
        direction: _busy_ns(
            (span.start_ns, span.end_ns)
            for span in result.transfers
            if span.direction == direction
        )
        for direction in ("fetch", "evict")
    }
    lane_bytes = {
        direction: sum(
            span.execution_bytes
            for span in result.transfers
            if span.direction == direction
        )
        for direction in ("fetch", "evict")
    }
    host_bytes = {
        direction: sum(
            span.bytes for span in result.transfers if span.direction == direction
        )
        for direction in ("fetch", "evict")
    }
    pool = result.execution
    return {
        "seconds": span_ns / 1e9,
        "tokens_per_second": (
            None if tokens_per_step is None else tokens_per_step / (span_ns / 1e9)
        ),
        "spill_peak_gib": result.spill.peak()[0] / GIB,
        "execution_peak_gib": None if pool is None else pool.peak()[0] / GIB,
        "idle_percent": 100.0 * max(span_ns - compute_ns, 0) / span_ns,
        "recompute_percent": (
            None if recompute_ns is None else 100.0 * recompute_ns / span_ns
        ),
        "fetch_utilization_percent": 100.0 * lane_ns["fetch"] / span_ns,
        "evict_utilization_percent": 100.0 * lane_ns["evict"] / span_ns,
        "fetch_gib": lane_bytes["fetch"] / GIB,
        "evict_gib": lane_bytes["evict"] / GIB,
        "fetch_host_gib": host_bytes["fetch"] / GIB,
        "evict_host_gib": host_bytes["evict"] / GIB,
        # the rate this view achieved, bytes over lane time, against the rate
        # the plan was priced at
        "fetch_gbps": (
            None if lane_ns["fetch"] == 0 else lane_bytes["fetch"] / lane_ns["fetch"]
        ),
        "evict_gbps": (
            None if lane_ns["evict"] == 0 else lane_bytes["evict"] / lane_ns["evict"]
        ),
        "assumed_fetch_gbps": _gbps(
            result.assumed.get("fetch_blended_bandwidth_bytes_per_second")
        ),
        "assumed_evict_gbps": _gbps(
            result.assumed.get("evict_blended_bandwidth_bytes_per_second")
        ),
        "assumed_fetch_solo_gbps": _gbps(
            result.assumed.get("fetch_solo_bandwidth_bytes_per_second")
        ),
        "assumed_fetch_concurrent_gbps": _gbps(
            result.assumed.get("fetch_concurrent_bandwidth_bytes_per_second")
        ),
        "assumed_evict_solo_gbps": _gbps(
            result.assumed.get("evict_solo_bandwidth_bytes_per_second")
        ),
        "assumed_evict_concurrent_gbps": _gbps(
            result.assumed.get("evict_concurrent_bandwidth_bytes_per_second")
        ),
        "assumed_fetch_latency_us": _us(result.assumed.get("fetch_latency_ns")),
        "assumed_evict_latency_us": _us(result.assumed.get("evict_latency_ns")),
    }


def _gbps(bytes_per_second: float | None) -> float | None:
    return None if bytes_per_second is None else bytes_per_second / 1e9


def _us(nanoseconds: float | None) -> float | None:
    return None if nanoseconds is None else nanoseconds / 1e3
