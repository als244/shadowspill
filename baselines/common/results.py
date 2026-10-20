"""The result a harness writes: one row per configuration that ran, under one
schema, with the measurement contract as code.

A row names the harness, the catalog model and mode, the precision, the
geometry, every harness setting that applied, the hardware, the provenance
(sources, packages, the kernels that ran) and what was measured: the median
over measured steps of the slowest rank's time, tokens per second, useful
TFLOP/s, device and host peaks. A configuration that ran out of device memory
is a row too (status ``cuda_out_of_memory``): it is a point on the frontier's
boundary, not a missing measurement.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The schema every row carries; a reader refuses another.
RESULT_SCHEMA = "shadowspill.baselines_result/v1"

#: The statuses a row may carry.
STATUSES = ("completed", "cuda_out_of_memory", "failed", "timeout", "interrupted")

#: What ``measurement`` holds for a completed row.
MEASUREMENT_KEYS = (
    "step_seconds",
    "global_tokens_per_second",
    "per_rank_cycle_seconds",
    "slowest_rank_cycle_seconds",
    "aggregation",
    "useful_tflops_per_gpu",
)


def summarize_cycles(rank_cycles, *, global_tokens, useful_flops_per_step=None):
    if not rank_cycles or not rank_cycles[0]:
        raise ValueError("no completed measurement cycles")
    width = len(rank_cycles[0])
    if any(len(row) != width for row in rank_cycles):
        raise ValueError("ranks completed different numbers of steps")
    if any(not math.isfinite(t) or t <= 0 for row in rank_cycles for t in row):
        raise ValueError("invalid cycle time")
    slowest = [max(values) for values in zip(*rank_cycles, strict=True)]
    seconds = statistics.median(slowest)
    return {
        "step_seconds": seconds,
        "global_tokens_per_second": global_tokens / seconds,
        "per_rank_cycle_seconds": rank_cycles,
        "slowest_rank_cycle_seconds": slowest,
        "aggregation": "median of per-step maximum rank time, excluding warmup",
        "useful_tflops_per_gpu": None
        if useful_flops_per_step is None
        else (useful_flops_per_step / seconds / len(rank_cycles) / 1e12),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def row(
    *,
    harness: str,
    status: str,
    model: Mapping[str, Any],
    geometry: Mapping[str, Any],
    precision: Mapping[str, Any],
    configuration: Mapping[str, Any],
    hardware: Mapping[str, Any],
    provenance: Mapping[str, Any],
    measurement: Mapping[str, Any] | None = None,
    memory: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One result row; ``validate_row`` is applied before it is returned."""

    result = {
        "schema": RESULT_SCHEMA,
        "harness": harness,
        "status": status,
        "model": dict(model),
        "geometry": dict(geometry),
        "precision": dict(precision),
        "configuration": dict(configuration),
        "hardware": dict(hardware),
        "provenance": dict(provenance),
        "measurement": None if measurement is None else dict(measurement),
        "memory": None if memory is None else dict(memory),
        "failure": None if failure is None else dict(failure),
    }
    validate_row(result)
    return result


def validate_row(value: Mapping[str, Any]) -> None:
    """Refuse a row that is not one of this schema, or a completed row that
    lacks its measurement or memory peaks."""

    if value.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} row: {value.get('schema')!r}")
    if value.get("status") not in STATUSES:
        raise ValueError(f"unknown row status {value.get('status')!r}")
    for key in (
        "harness",
        "model",
        "geometry",
        "precision",
        "configuration",
        "hardware",
        "provenance",
    ):
        if key not in value:
            raise ValueError(f"row lacks {key!r}")
    if not value["model"].get("model"):
        raise ValueError("row.model must name the catalog model")
    if value["status"] == "completed":
        measurement = value.get("measurement") or {}
        missing = [key for key in MEASUREMENT_KEYS if key not in measurement]
        if missing:
            raise ValueError(f"completed row lacks measurement keys {missing}")
        memory = value.get("memory") or {}
        for key in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "host_peak_rss_bytes",
        ):
            if key not in memory:
                raise ValueError(f"completed row lacks memory.{key}")
    elif value["status"] in ("failed", "timeout") and not value.get("failure"):
        raise ValueError(f"a {value['status']} row carries its failure")


def memory_peaks(rank_results: list[Mapping[str, Any]]) -> dict[str, int]:
    """Device peaks are the maximum over ranks (the binding device); the host
    peak is the sum of per-process high-water marks, not a simultaneous
    node measurement, and says so in its name."""

    return {
        "peak_allocated_bytes": max(
            int(r["peak_allocated_bytes"]) for r in rank_results
        ),
        "peak_reserved_bytes": max(int(r["peak_reserved_bytes"]) for r in rank_results),
        "host_peak_rss_bytes": sum(int(r["host_peak_rss_bytes"]) for r in rank_results),
        "host_peak_rss_aggregation": "sum of per-process high-water marks",
    }


def read_rows(root: Path) -> list[dict[str, Any]]:
    """Every row under ``root``: each ``row.json`` a harness or sweep wrote."""

    rows = []
    for path in sorted(Path(root).rglob("row.json")):
        value = json.loads(path.read_text())
        validate_row(value)
        value["_path"] = str(path)
        rows.append(value)
    return rows
