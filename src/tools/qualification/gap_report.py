"""Real-versus-simulated gap report over a performance matrix's traced warm steps.

Reads each cell's saved step diagnostics (``shadowspill.step_diagnostics``)
and its PressureFit fixture, and prints, per model, where the step's time went
against the simulation: the span, task, and idle deltas; task-duration error
by phase; task start drift along the compute lane; each transfer lane's
assumed versus effective bandwidth; and every measured transfer's achieved
rate classified by how much of its device interval overlapped the opposite
lane -- solo, mixed, or concurrent -- beside the run's own solo and concurrent
calibration figures.

This is the acceptance experiment for simulator changes: run the performance
matrix, then this report, and the lane-busy, drift, and error numbers say
whether the model got closer to the hardware.

    python -m tools.qualification.gap_report qualification/results/<matrix directory>
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shadowspill.schema import artifact_schema

_LARGE_COPY_BYTES = 64 << 20
_SOLO_OVERLAP = 0.05
_CONCURRENT_OVERLAP = 0.95


def _overlap_fraction(
    start: float, end: float, other: Sequence[tuple[float, float]]
) -> float:
    """Share of [start, end) covered by the sorted, disjoint intervals in other."""

    if end <= start:
        return 0.0
    covered = 0.0
    index = max(0, bisect.bisect_left(other, (start, -1.0)) - 1)
    while index < len(other) and other[index][0] < end:
        low = max(other[index][0], start)
        high = min(other[index][1], end)
        if high > low:
            covered += high - low
        index += 1
    return covered / (end - start)


def _quantile_text(values: Sequence[float]) -> str:
    ordered = sorted(values)
    if len(ordered) >= 10:
        deciles = statistics.quantiles(ordered, n=10)
        low, high = deciles[0], deciles[-1]
    else:
        low, high = ordered[0], ordered[-1]
    return f"median {statistics.median(ordered):5.1f}  p10 {low:5.1f}  p90 {high:5.1f}"


def _calibration(cell: Mapping[str, Any]) -> dict[str, tuple[float, float, float]]:
    profiles: dict[str, tuple[float, float, float]] = {}
    for profile in cell["runtime_transfer_capabilities"]["profiles"]:
        if profile["calibration_mode"] == "identity":
            continue
        direction = "fetch" if profile["source"] == "spill" else "evict"
        profiles[direction] = (
            profile["solo_bandwidth_bytes_per_second"] / 1e9,
            profile["concurrent_bandwidth_bytes_per_second"] / 1e9,
            profile["bandwidth_bytes_per_second"] / 1e9,
        )
    return profiles


def _milliseconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1e3:+.0f} ms"


_SIZE_BUCKETS: tuple[tuple[str, int], ...] = (
    ("< 1 MB", 1 << 20),
    ("1-16 MB", 16 << 20),
    ("16-64 MB", 64 << 20),
    ("64-256 MB", 256 << 20),
    ("256 MB-1 GB", 1 << 30),
    (">= 1 GB", 1 << 62),
)


def _lane_interval(record: Mapping[str, Any]) -> tuple[float, float]:
    """When the copy started and finished on its lane."""

    lane = record["lane"]
    return lane["lane_started_at_seconds"], lane["lane_finished_at_seconds"]


def _lane_duration(record: Mapping[str, Any]) -> float:
    """How long the copy held its lane."""

    started, finished = _lane_interval(record)
    return float(finished - started)


def _simulated_duration(record: Mapping[str, Any]) -> float:
    """The lane time the simulator priced for the copy."""

    simulated = record["simulated"]
    started = simulated["simulated_started_at_seconds"]
    finished = simulated["simulated_finished_at_seconds"]
    if started is None or finished is None:
        return 0.0
    return float(finished - started)


def _bucket(size: int) -> str:
    for label, limit in _SIZE_BUCKETS:
        if size < limit:
            return label
    return _SIZE_BUCKETS[-1][0]


def _print_size_buckets(
    direction: str,
    records: Sequence[Mapping[str, Any]],
    other: Sequence[tuple[float, float]],
) -> None:
    """Achieved rate per (size bucket, overlap class), every measured transfer."""

    groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    total_bytes = sum(item["bytes"] for item in records)
    for item in records:
        start, end = _lane_interval(item)
        if end <= start:
            continue
        fraction = _overlap_fraction(start, end, other)
        kind = (
            "solo"
            if fraction < _SOLO_OVERLAP
            else "concurrent"
            if fraction > _CONCURRENT_OVERLAP
            else "mixed"
        )
        groups.setdefault((_bucket(item["bytes"]), kind), []).append(
            (item["bytes"], item["bytes"] / (end - start) / 1e9)
        )
    print(f"    {direction} by size and overlap (share of lane bytes, achieved GB/s):")
    for label, _limit in _SIZE_BUCKETS:
        for kind in ("solo", "mixed", "concurrent"):
            values = groups.get((label, kind))
            if not values:
                continue
            share = 100 * sum(size for size, _rate in values) / max(total_bytes, 1)
            rates = [rate for _size, rate in values]
            print(
                f"      {label:12s} {kind:10s} n={len(values):5d}"
                f"  {share:5.1f}% of bytes  {_quantile_text(rates)}"
            )


def _measured(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        item for item in records if item["lane"]["lane_started_at_seconds"] is not None
    ]


def _print_step(cell: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> None:
    summary = diagnostics["summary"]
    timelines = diagnostics["timelines"]
    tasks = diagnostics["tasks"]
    print(
        f"  span: simulated {summary['simulated_selected_span_seconds']:.3f} s"
        f"  real {summary['real_selected_span_seconds']:.3f} s"
        f"  delta {summary['selected_span_delta_seconds']:+.3f} s"
        f" | task durations {summary['task_event_delta_seconds']:+.3f} s"
        f" | idle simulated {summary['simulated_inter_task_idle_seconds']:.3f} s"
        f" real {summary['real_inter_task_idle_seconds']:.3f} s"
        f" (waiting {summary['real_inter_task_readiness_wait_seconds']:.3f} s,"
        f" exposed {summary['real_inter_task_exposed_overhead_seconds'] * 1e3:.1f} ms)"
        f" | first kernel at"
        f" {timelines['clocks']['first_task_started_at_seconds'] * 1e3:.0f} ms,"
        f" of which input wait"
        f" {summary['real_initial_readiness_wait_seconds'] * 1e3:.0f} ms"
    )
    phases = summary["phase_comparisons"]
    print(
        "  task duration delta by phase: "
        + ", ".join(
            f"{phase} {item['delta_seconds']:+.3f} s"
            f" ({100 * item['delta_seconds'] / max(item['profiled_task_seconds'], 1e-9):+.1f}%)"  # noqa: E501
            for phase, item in phases.items()
        )
    )
    starts = [tasks[key]["delta"]["start_seconds"] for key in timelines["compute"]]
    print(
        f"  task start drift: first {starts[0] * 1e3:+.1f} ms,"
        f" median {statistics.median(starts) * 1e3:+.1f} ms,"
        f" last {starts[-1] * 1e3:+.1f} ms, most negative {min(starts) * 1e3:+.1f} ms"
    )


def _print_lanes(cell: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> None:
    timelines = diagnostics["timelines"]
    transfers = diagnostics["transfers"]
    calibration = _calibration(cell)
    intervals = {
        direction: sorted(
            _lane_interval(item) for item in _measured(transfers[direction].values())
        )
        for direction in ("fetch", "evict")
    }
    for direction, other in (("fetch", "evict"), ("evict", "fetch")):
        lane = timelines[direction]["summary"]
        records = _measured(transfers[direction].values())
        scheduled = [item for item in records if item["triggered_by"] != "init"]
        solo_rate, concurrent_rate, planned_rate = calibration[direction]
        effective = lane["effective_bandwidth_bytes_per_second"]
        ratios = [
            _lane_duration(item) / _simulated_duration(item)
            for item in scheduled
            if item["bytes"] >= _LARGE_COPY_BYTES and _simulated_duration(item)
        ]
        print(
            f"  {direction}: {len(scheduled)} scheduled, {lane['opening_transfers']}"
            f" opening ({lane['bytes'] / 1e9:.1f} GB in all), measured {len(records)}"
            f" | planned {planned_rate:.1f} GB/s, effective"
            f" {effective / 1e9 if effective else float('nan'):.1f} GB/s"
            f" | copy duration real/sim median"
            f" {statistics.median(ratios) if ratios else float('nan'):.2f}"
            f" | lane busy simulated {lane['simulated_busy_seconds']:.3f} s"
            f" real {lane['lane_busy_seconds']:.3f} s"
            f" | largest start delta"
            f" {_milliseconds(lane['largest_start_delta_seconds'])}"
            f" at {lane['largest_start_delta_transfer_id']}"
        )
        classes: dict[str, list[float]] = {"solo": [], "mixed": [], "concurrent": []}
        for item in records:
            start, end = _lane_interval(item)
            if item["bytes"] < _LARGE_COPY_BYTES or end <= start:
                continue
            fraction = _overlap_fraction(start, end, intervals[other])
            kind = (
                "solo"
                if fraction < _SOLO_OVERLAP
                else "concurrent"
                if fraction > _CONCURRENT_OVERLAP
                else "mixed"
            )
            classes[kind].append(item["bytes"] / (end - start) / 1e9)
        print(
            f"    achieved rate by overlap with the {other} lane (calibration solo"
            f" {solo_rate:.1f}, concurrent {concurrent_rate:.1f} GB/s):"
        )
        for kind, values in classes.items():
            if values:
                print(f"      {kind:10s} n={len(values):5d}  {_quantile_text(values)}")
            else:
                print(f"      {kind:10s} n=    0")
        _print_size_buckets(direction, records, intervals[other])


def report(directory: Path) -> int:
    cells = sorted(directory.glob("mlops_*.json"))
    if not cells:
        raise SystemExit(f"no performance cells under {directory}")
    for path in cells:
        cell = json.loads(path.read_text())
        diagnostics = cell["warm_diagnostics"]
        if diagnostics.get("schema") != artifact_schema("step_diagnostics"):
            raise SystemExit(
                f"{path} carries {diagnostics.get('schema')}, not this version"
            )
        print(
            f"=== {path.stem}: simulated makespan"
            f" {diagnostics['summary']['simulator_makespan_seconds']:.3f} s,"
            f" real step {cell['median_step_seconds']:.3f} s,"
            f" error {100 * cell['simulator_relative_error']:+.2f}% ==="
        )
        _print_step(cell, diagnostics)
        _print_lanes(cell, diagnostics)
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "directory", type=Path, help="a performance matrix output directory"
    )
    arguments = parser.parse_args()
    return report(arguments.directory)


if __name__ == "__main__":
    raise SystemExit(main())
