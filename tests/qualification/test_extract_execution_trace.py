from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qualification.extract_execution_trace import extract_trace


def _create_trace(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, text TEXT, globalTid INTEGER,
                textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, correlationId INTEGER,
                globalTid INTEGER, nameId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, streamId INTEGER,
                correlationId INTEGER, shortName INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER, streamId INTEGER,
                correlationId INTEGER, copyKind INTEGER
            );
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, label TEXT);

            INSERT INTO StringIds VALUES (1, 'kernel');
            INSERT INTO StringIds VALUES (2, 'cudaLaunchKernel');
            INSERT INTO StringIds VALUES (3, 'cudaMemcpyAsync');
            INSERT INTO StringIds VALUES (4, 'cudaEventRecord');
            INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'Host-to-Device');

            INSERT INTO NVTX_EVENTS VALUES
              (0, 1000, 'shadowspill.task.forward.task_000000', 10, NULL),
              (0, 100, 'shadowspill.before_task.task_000000', 10, NULL),
              (100, 200, 'shadowspill.storage_rebind.task_000000', 10, NULL),
              (200, 600, 'shadowspill.compiled_call.task_000000', 10, NULL),
              (600, 900, 'shadowspill.after_task.task_000000', 10, NULL),
              (1000, 2000, 'shadowspill.task.optimizer.task_000001', 10, NULL),
              (1000, 1100, 'shadowspill.before_task.task_000001', 10, NULL),
              (1100, 1200, 'shadowspill.storage_rebind.task_000001', 10, NULL),
              (1200, 1600, 'shadowspill.compiled_call.task_000001', 10, NULL),
              (1600, 1900, 'shadowspill.after_task.task_000001', 10, NULL),
              (700, 900, 'shadowspill.runtime.transfer.h2d', 20, NULL);

            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
              (250, 260, 1, 10, 2),
              (1250, 1260, 2, 10, 2),
              (750, 760, 3, 20, 3),
              (50, 51, 4, 10, 4);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
              (300, 500, 7, 1, 1),
              (1300, 1500, 7, 2, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
              (780, 880, 8, 3, 1);
            """
        )


def test_extracts_tasks_optimizer_idle_and_transfers(tmp_path: Path) -> None:
    trace = tmp_path / "trace.sqlite"
    _create_trace(trace)

    result = extract_trace(trace)

    assert result["task_count"] == 2
    assert result["optimizer_task_count"] == 1
    assert result["compute_stream_id"] == 7
    assert result["compute_span_ns"] == 1200
    assert result["compute_kernel_union_ns"] == 400
    assert result["compute_idle_ns"] == 800
    assert result["optimizer_span_ns"] == 200
    assert result["phase_kernel_ns"] == {"forward": 200, "optimizer": 200}
    assert result["transfer_dispatch_ns"] == {"h2d": 200}
    assert result["transfers"] == {
        "Host-to-Device": {
            "count": 1,
            "duration_ns": 100,
            "compute_overlap_ns": 0,
        }
    }
    assert result["cuda_api_counts"] == {"cudaEventRecord": 1}
    tasks = result["tasks"]
    assert isinstance(tasks, list)
    first = tasks[0]
    assert isinstance(first, dict)
    assert first["task_id"] == "task_000000"
    assert first["host_segments_ns"] == {
        "before_task": 100,
        "storage_rebind": 100,
        "compiled_call": 400,
        "after_task": 300,
    }


def test_rejects_duplicate_semantic_task_ranges(tmp_path: Path) -> None:
    trace = tmp_path / "duplicate.sqlite"
    _create_trace(trace)
    name = (
        "shadowspill.pytorch.task.execution_000000."
        "microbatch_0000.stage_0000.forward.recompute"
    )
    with sqlite3.connect(trace) as connection:
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)",
            ((0, 1000, name, 10), (1, 999, name, 10)),
        )

    with pytest.raises(ValueError, match="duplicate semantic task range"):
        extract_trace(trace)
