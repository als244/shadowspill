"""A profiler subprocess exit alone cannot prove metrics or all-rank coverage."""

import copy
import sqlite3
from types import SimpleNamespace

import pytest

from tools.pressurefit_trace.profile_winners import (
    execution_structure,
    replay_request_resolver,
    require_exact_plan,
    validate_sqlite,
)


def replay_program():
    return {
        "tasks": [{"task_id": "backward", "inputs": ["weight"]}],
        "objects": [{"object_id": "weight", "size_bytes": 128}],
        "profiles": [
            {
                "profile_id": "p",
                "runtime_ns": 10,
                "workspace_bytes": 64,
                "compatibility_digest": "original",
                "overlap_costs": [],
                "resource_phases": [{"resource": "nccl", "duration_ns": 5}],
            }
        ],
        "alias_groups": [
            {
                "alias_group_id": "a",
                "size_bytes": 128,
                "transfer": {
                    "spill_size_bytes": 16,
                    "evict": [],
                    "fetch": [
                        {"kind": "copy", "bytes": 16, "duration_ns": 0},
                        {
                            "kind": "timed",
                            "bytes": 112,
                            "duration_ns": 5,
                            "resource": "nccl",
                            "submission_ordered": True,
                        },
                    ],
                },
            }
        ],
    }


def test_replay_distinguishes_changed_costs_from_changed_execution_contracts():
    original = replay_program()
    measured = copy.deepcopy(original)
    measured["profiles"][0].update(
        runtime_ns=20, compatibility_digest="new", overlap_costs=[{"task_rate": 0.5}]
    )
    measured["profiles"][0]["resource_phases"][0]["duration_ns"] = 9
    measured["alias_groups"][0]["transfer"]["fetch"][1]["duration_ns"] = 11
    assert measured != original
    assert execution_structure(measured) == execution_structure(original)
    assert original["profiles"][0]["runtime_ns"] == 10


@pytest.mark.parametrize(
    "change", ["workspace", "task", "shard", "collective", "resource"]
)
def test_replay_never_treats_storage_or_communication_changes_as_timing(change):
    original = replay_program()
    changed = copy.deepcopy(original)
    if change == "workspace":
        changed["profiles"][0]["workspace_bytes"] += 1
    elif change == "task":
        changed["tasks"][0]["inputs"] = []
    elif change == "shard":
        changed["alias_groups"][0]["transfer"]["spill_size_bytes"] *= 2
    elif change == "collective":
        changed["alias_groups"][0]["transfer"]["fetch"][1]["bytes"] *= 2
    else:
        changed["profiles"][0]["resource_phases"][0]["resource"] = "different"
    assert execution_structure(changed) != execution_structure(original)


def test_replay_preserves_admission_policy_without_changing_calibration_or_budget():
    request = {"plan_store_mode": "contribute"}
    arguments = SimpleNamespace(
        plan_store_mode="require",
        artifact_store_mode="require",
        transfer_bandwidths=object(),
        search_budget_gib=[20, 30],
    )
    calibration = arguments.transfer_bandwidths
    resolved = replay_request_resolver(lambda parser, values: values, request)(
        None, arguments
    )
    assert resolved.plan_store_mode == "contribute"
    assert resolved.artifact_store_mode == "require"
    assert resolved.transfer_bandwidths is calibration
    assert resolved.search_budget_gib == [20, 30]


@pytest.mark.parametrize(
    "field", ["program_digest", "schedule_digest", "layout_digest", "capture_identity"]
)
def test_replay_rejects_any_changed_plan_identity(field):
    expected = dict.fromkeys(
        ("program_digest", "schedule_digest", "layout_digest", "capture_identity"),
        "original",
    )
    require_exact_plan(expected, expected)
    with pytest.raises(ValueError, match=field):
        require_exact_plan({**expected, field: "changed"}, expected)


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
                ("shadowspill.pytorch.task.execution_0",),
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
