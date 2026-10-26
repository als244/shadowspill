"""Validate GPU, device-metric and semantic-range coverage in an Nsight export."""

import re
import sqlite3
from pathlib import Path


def validate_sqlite(path, *, world_size, capture_steps=None, warmup_steps=3):
    """Require actual CUDA, NVTX and device-metric rows, not only an exit code."""
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as database:
        tables = {
            row[0]
            for row in database.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        return {
            "status": "validated",
            "rows": activity_counts(database, tables),
            "metric_groups": metric_groups(database, world_size),
            "kernel_devices": kernel_devices(database, tables, world_size),
            **semantic_ranges(
                database, tables, world_size, capture_steps, warmup_steps
            ),
        }


def activity_counts(database, tables):
    counts = {}
    for name in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "NVTX_EVENTS",
        "GPU_METRICS",
    ):
        counts[name] = (
            database.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            if name in tables
            else 0
        )
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Nsight capture lacks requested evidence: {counts}")
    return counts


def metric_groups(database, world_size):
    columns = {row[1] for row in database.execute('PRAGMA table_info("GPU_METRICS")')}
    field = "typeId" if "typeId" in columns else "gpuId" if "gpuId" in columns else None
    if field is None:
        raise ValueError(
            "unknown GPU metric schema; inspect GPU_METRICS before accepting"
        )
    groups = list(
        database.execute(
            f'SELECT "{field}", count(*) FROM GPU_METRICS GROUP BY "{field}"'
        )
    )
    if len(groups) < world_size:
        raise ValueError(f"metrics do not cover all {world_size} GPUs: {groups}")
    return groups


def kernel_devices(database, tables, world_size):
    if "TARGET_INFO_CUDA_DEVICE" in tables:
        # CUDA ordinals are process-local: device_worker exposes each rank's
        # physical GPU as cuda:0. Resolve the process/ordinal pair before
        # checking physical GPU coverage. Nsight packs PID into bits 24:48.
        kernel_devices = list(
            database.execute(
                "SELECT d.gpuId, count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "LEFT JOIN TARGET_INFO_CUDA_DEVICE d "
                "ON d.pid=((k.globalPid >> 24) & 0xffffff) "
                "AND d.cudaId=k.deviceId GROUP BY d.gpuId"
            )
        )
        if any(device is None for device, _ in kernel_devices):
            raise ValueError("cannot map CUDA kernels to physical GPUs")
    else:
        kernel_devices = list(
            database.execute(
                "SELECT deviceId, count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "GROUP BY deviceId"
            )
        )
    if len(kernel_devices) < world_size:
        raise ValueError(
            f"CUDA kernels do not cover all {world_size} GPUs: {kernel_devices}"
        )
    return kernel_devices


def semantic_ranges(database, tables, world_size, capture_steps, warmup_steps):
    text_columns = {
        row[1] for row in database.execute('PRAGMA table_info("NVTX_EVENTS")')
    }
    if (
        "text" not in text_columns
        or "textId" not in text_columns
        or "StringIds" not in tables
    ):
        raise ValueError("unknown NVTX text schema; inspect before accepting")
    labels = dict(
        database.execute(
            "SELECT coalesce(n.text, s.value), count(*) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId=s.id "
            "GROUP BY coalesce(n.text,s.value)"
        )
    )
    task_ranges = sum(
        count
        for name, count in labels.items()
        if name and name.startswith(("shadowspill.task.", "shadowspill.pytorch.task."))
    )
    transfer_ranges = sum(
        count
        for name, count in labels.items()
        if name and name.startswith("shadowspill.runtime.transfer.")
    )
    steps = {
        name: count
        for name, count in labels.items()
        if name and name.startswith("shadowspill.capture/")
    }
    step_ranks = {
        int(found.group(1))
        for name in steps
        if (found := re.search(r"/rank-(\d+)/step-", name)) is not None
    }
    if not task_ranges or not transfer_ranges or step_ranks != set(range(world_size)):
        raise ValueError(
            "Nsight capture lacks semantic task/transfer/all-rank step ranges"
        )
    if capture_steps is not None:
        observed = {
            (int(match.group(1)), int(match.group(2)))
            for name, count in steps.items()
            if count == 1 and (match := re.search(r"/rank-(\d+)/step-(\d+)$", name))
        }
        expected = {
            (rank, step)
            for rank in range(world_size)
            for step in range(warmup_steps + 1, warmup_steps + capture_steps + 1)
        }
        if observed != expected:
            raise ValueError("Nsight capture does not contain every requested step")
    return {
        "task_ranges": task_ranges,
        "transfer_ranges": transfer_ranges,
        "step_ranges": steps,
    }
