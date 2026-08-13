"""Extract per-invocation timing from a profile-variance NSYS SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_RANGE_PREFIX = "shadowspill.qualification.profile_variance."


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _extract(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    ranges = connection.execute(
        """
        SELECT n.start, n.end, n.globalTid,
               COALESCE(n.text, strings.value) AS name
          FROM NVTX_EVENTS AS n
          LEFT JOIN StringIds AS strings ON strings.id = n.textId
         WHERE COALESCE(n.text, strings.value) LIKE ?
           AND n.end IS NOT NULL
         ORDER BY n.start
        """,
        (f"{_RANGE_PREFIX}%",),
    ).fetchall()
    invocations: list[dict[str, Any]] = []
    for item in ranges:
        runtime_calls = connection.execute(
            """
            SELECT api.value AS name, runtime.start, runtime.end,
                   runtime.correlationId
              FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
              JOIN StringIds AS api ON api.id = runtime.nameId
             WHERE runtime.globalTid = ?
               AND runtime.start >= ? AND runtime.end <= ?
             ORDER BY runtime.start
            """,
            (item["globalTid"], item["start"], item["end"]),
        ).fetchall()
        correlation_ids = [
            call["correlationId"]
            for call in runtime_calls
            if call["correlationId"] is not None
        ]
        kernels: list[sqlite3.Row] = []
        if correlation_ids:
            placeholders = ",".join("?" for _ in correlation_ids)
            kernels = connection.execute(
                f"""
                SELECT kernel.start, kernel.end, kernel.streamId,
                       names.value AS name
                  FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
                  JOIN StringIds AS names ON names.id = kernel.shortName
                 WHERE kernel.correlationId IN ({placeholders})
                 ORDER BY kernel.start
                """,
                correlation_ids,
            ).fetchall()
        api_ns: Counter[str] = Counter()
        for call in runtime_calls:
            api_ns[call["name"]] += call["end"] - call["start"]
        kernel_ns = sum(kernel["end"] - kernel["start"] for kernel in kernels)
        kernel_envelope_ns = (
            max(kernel["end"] for kernel in kernels)
            - min(kernel["start"] for kernel in kernels)
            if kernels
            else 0
        )
        nested = connection.execute(
            """
            SELECT COALESCE(n.text, strings.value) AS name,
                   n.end - n.start AS duration
              FROM NVTX_EVENTS AS n
              LEFT JOIN StringIds AS strings ON strings.id = n.textId
             WHERE n.globalTid = ? AND n.start >= ? AND n.end <= ?
               AND COALESCE(n.text, strings.value) LIKE 'shadowspill.runtime.%'
            """,
            (item["globalTid"], item["start"], item["end"]),
        ).fetchall()
        nested_ns: Counter[str] = Counter()
        nested_calls: Counter[str] = Counter()
        for event in nested:
            nested_ns[event["name"]] += event["duration"]
            nested_calls[event["name"]] += 1
        invocations.append(
            {
                "name": item["name"],
                "kind": item["name"].split(".")[-2],
                "host_range_ns": item["end"] - item["start"],
                "kernel_count": len(kernels),
                "kernel_sum_ns": kernel_ns,
                "kernel_envelope_ns": kernel_envelope_ns,
                "compute_stream_gap_ns": max(0, kernel_envelope_ns - kernel_ns),
                "cuda_api_sum_ns": sum(api_ns.values()),
                "cuda_api_ns": dict(api_ns),
                "runtime_annotation_ns": dict(nested_ns),
                "runtime_annotation_calls": dict(nested_calls),
            }
        )
    summary: dict[str, Any] = {}
    for kind in ("zero", "normal"):
        selected = [item for item in invocations if item["kind"] == kind]
        annotations: dict[str, list[int]] = defaultdict(list)
        annotation_calls: Counter[str] = Counter()
        for item in selected:
            for name, duration in item["runtime_annotation_ns"].items():
                annotations[name].append(duration)
            annotation_calls.update(item["runtime_annotation_calls"])
        summary[kind] = {
            "invocations": len(selected),
            "median_host_range_ns": _median(
                [item["host_range_ns"] for item in selected]
            ),
            "median_kernel_sum_ns": _median(
                [item["kernel_sum_ns"] for item in selected]
            ),
            "median_kernel_envelope_ns": _median(
                [item["kernel_envelope_ns"] for item in selected]
            ),
            "median_compute_stream_gap_ns": _median(
                [item["compute_stream_gap_ns"] for item in selected]
            ),
            "median_cuda_api_sum_ns": _median(
                [item["cuda_api_sum_ns"] for item in selected]
            ),
            "runtime_annotations": {
                name: {
                    "total_calls": annotation_calls[name],
                    "median_total_ns_per_invocation": _median(values),
                }
                for name, values in sorted(annotations.items())
            },
        }
    connection.close()
    return {
        "schema": "shadowspill.profile_variance_nsys/v1",
        "source": str(database),
        "summary": summary,
        "invocations": invocations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = _extract(arguments.database)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
