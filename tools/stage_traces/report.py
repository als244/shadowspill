"""Verify that a report actually captured kernels and the requested NVTX ranges."""

from __future__ import annotations

import sqlite3
from collections import Counter


def verify(path, iterations, *, gpu_uuid=None):
    with sqlite3.connect(path) as db:
        tables = {
            r[0]
            for r in db.execute("select name from sqlite_master where type='table'")
        }
        if "NVTX_EVENTS" not in tables:
            raise ValueError("trace contains no NVTX events")
        strings = (
            dict(db.execute("select id, value from StringIds"))
            if "StringIds" in tables
            else {}
        )
        rows = db.execute("select * from NVTX_EVENTS")
        names = [r[0] for r in rows.description]
        labels = []
        for row in rows:
            event = dict(zip(names, row, strict=True))
            if event.get("end") is None:
                continue
            label = event.get("text") or strings.get(event.get("textId"))
            if label:
                labels.append(label)
        counts = Counter(labels)
        if counts["forward"] != iterations or counts["backward"] != iterations:
            raise ValueError(f"wrong phase-range counts: {dict(counts)}")
        for i in range(iterations):
            if counts[f"iteration/{i}"] != 1:
                raise ValueError(f"missing/duplicate iteration/{i}")
        kernels = sum(
            db.execute(f'select count(*) from "{table}"').fetchone()[0]
            for table in tables
            if table.startswith("CUPTI_ACTIVITY_KIND_KERNEL")
        )
        if kernels == 0:
            raise ValueError("trace contains no CUDA kernels")
        result = {"nvtx_ranges": dict(counts), "cuda_kernel_count": kernels}
        if gpu_uuid is not None:
            required = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS", "TARGET_INFO_GPU"}
            if missing := required - tables:
                raise ValueError(f"missing GPU metrics tables: {sorted(missing)}")
            samples = list(
                db.execute(
                    "select typeId, count(*), count(distinct timestamp), "
                    "min(timestamp), max(timestamp) from GPU_METRICS group by typeId"
                )
            )
            if len(samples) != 1:
                raise ValueError(f"expected metrics from one GPU, found {len(samples)}")
            type_id, rows, timestamps, first, last = samples[0]
            # NVIDIA's bundled nsys_recipe/recipes/nvlink_sum uses these
            # low eight bits to join GPU_METRICS with TARGET_INFO_GPU.id.
            physical_id = type_id & 0xFF
            devices = list(
                db.execute(
                    "select uuid, busLocation, name from TARGET_INFO_GPU where id=?",
                    (physical_id,),
                )
            )
            expected = gpu_uuid.lower().removeprefix("gpu-")
            if (
                len(devices) != 1
                or devices[0][0].lower().removeprefix("gpu-") != expected
            ):
                raise ValueError(
                    f"GPU metrics device mismatch: physical id={physical_id}, "
                    f"devices={devices}, worker UUID={gpu_uuid}"
                )
            names = [
                r[0]
                for r in db.execute(
                    "select distinct metricName from TARGET_INFO_GPU_METRICS "
                    "where typeId=? order by metricName",
                    (type_id,),
                )
            ]
            if not names or not timestamps:
                raise ValueError("GPU metric definitions or samples are empty")
            result["gpu_metrics"] = {
                "physical_gpu_id": physical_id,
                "uuid": devices[0][0],
                "pci_bus_id": devices[0][1],
                "device_name": devices[0][2],
                "metric_names": names,
                "metric_count": len(names),
                "rows": rows,
                "sample_timestamps": timestamps,
                "first_timestamp_ns": first,
                "last_timestamp_ns": last,
            }
        return result


def write_index(output, cases, outcomes):
    """Portable index of every requested trace, including pending and OOM cases."""
    import json
    from pathlib import Path

    output = Path(output)
    by_directory = {row["case"]["directory"]: row for row in outcomes}
    lines = [
        "# Isolated compiled stage traces",
        "",
        "Unique forward/backward graph pairs, excluding embeddings and optimizers. "
        "Each trace uses one GPU; workers drain the shared queue. "
        "Forward and backward timings below include Nsight instrumentation.",
        "",
        "See [queue.json](queue.json), [run-manifest.json](run-manifest.json), "
        "and [progress.json](progress.json) for complete metadata.",
        "",
        "| Stage | Sequence | Batch | Variant | Status | FWD / BWD ms | Report | Log |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for case in cases:
        row = by_directory.get(case["directory"])
        status, timing, report, log = "pending", "", "", ""
        if row:
            status = row["status"]
            attempt = Path(row["case"]["attempt"])
            relative = attempt.relative_to(output).as_posix()
            log = f"[console]({relative}/logs/console.log)"
            if (attempt / "nsys/trace.nsys-rep").exists():
                report = f"[nsys-rep]({relative}/nsys/trace.nsys-rep)"
            result = attempt / "result.json"
            if result.exists():
                samples = json.loads(result.read_text())["samples"]
                if samples:
                    timing = " / ".join(
                        f"{sum(sample[field] for sample in samples) / len(samples):.3f}"
                        for field in ("forward_ms", "backward_ms")
                    )
        lines.append(
            f"| {case['stage']} | {case['sequence_length']} | {case['batch_size']} | "
            f"{case['variant']} | {status} | {timing} | {report} | {log} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")
