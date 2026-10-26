"""A profiler subprocess exit alone cannot prove metrics or all-rank coverage."""

import sqlite3

import pytest

from tools.pressurefit_trace.profile_winners.evidence import validate_sqlite


def evidence(path, *, metric_groups=2, kernel_devices=2, step_ranks=(0, 1)):
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (deviceId INTEGER);
            CREATE TABLE GPU_METRICS (typeId INTEGER);
            CREATE TABLE NVTX_EVENTS (text TEXT, textId INTEGER);
            CREATE TABLE StringIds (id INTEGER, value TEXT);
        """)
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)",
            [(i,) for i in range(kernel_devices)],
        )
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0)")
        db.executemany(
            "INSERT INTO GPU_METRICS VALUES (?)", [(i,) for i in range(metric_groups)]
        )
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,NULL)",
            [
                (f"shadowspill.capture/budget-20GiB/rank-{r}/step-{s}",)
                for r in step_ranks
                for s in (4, 5, 6)
            ]
            + [
                ("shadowspill.task.execution_0",),
                ("shadowspill.runtime.transfer.fetch/example",),
            ],
        )


def test_accepts_only_complete_all_rank_profiler_evidence(tmp_path):
    path = tmp_path / "capture.sqlite"
    evidence(path)
    assert validate_sqlite(path, world_size=2, capture_steps=3)["status"] == "validated"


@pytest.mark.parametrize("physical_devices", [(3, 5), (3, 3), (3, None)])
def test_resolves_rank_local_cuda_ordinals(tmp_path, physical_devices):
    path = tmp_path / "capture.sqlite"
    evidence(path)
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD globalPid INTEGER")
        db.execute(
            "CREATE TABLE TARGET_INFO_CUDA_DEVICE "
            "(gpuId INTEGER, cudaId INTEGER, pid INTEGER)"
        )
        db.executemany(
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=0, globalPid=? "
            "WHERE rowid=?",
            [
                ((1 << 48) | (pid << 24), index + 1)
                for index, pid in enumerate((41, 42))
            ],
        )
        db.executemany(
            "INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (?,0,?)",
            [
                (device, pid)
                for device, pid in zip(physical_devices, (41, 42), strict=True)
            ],
        )
    if physical_devices == (3, 5):
        result = validate_sqlite(path, world_size=2, capture_steps=3)
        assert result["kernel_devices"] == [(3, 1), (5, 1)]
    else:
        with pytest.raises(ValueError):
            validate_sqlite(path, world_size=2, capture_steps=3)


def test_rejects_missing_captured_steps(tmp_path):
    path = tmp_path / "capture.sqlite"
    evidence(path)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM NVTX_EVENTS WHERE text LIKE '%rank-1/step-6'")
    with pytest.raises(ValueError, match="every requested step"):
        validate_sqlite(path, world_size=2, capture_steps=3)


@pytest.mark.parametrize(
    "missing",
    [
        {"metric_groups": 1},
        {"kernel_devices": 1},
        {"step_ranks": (0,)},
    ],
)
def test_rejects_partial_profiler_evidence(tmp_path, missing):
    path = tmp_path / "capture.sqlite"
    evidence(path, **missing)
    with pytest.raises(ValueError):
        validate_sqlite(path, world_size=2)
