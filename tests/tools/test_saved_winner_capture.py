"""Capture selection uses the exact completed archive; no recapture or replanning."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarking.quickstart.phases.common import plan_identity, read, write
from shadowspill import MemoryBudgets, Plan, plan_program
from tests.shadowspill.offline._example import communication, profiled
from tools.pressurefit_trace.profile_winners.capture import command, supported_traces
from tools.pressurefit_trace.profile_winners.selection import (
    completed,
    digest,
    fingerprint,
    verified_capture,
)


@pytest.fixture
def saved_winner(tmp_path):
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(200, 64),
        communication_info=communication(),
    )
    winner = {
        "file": "budget-0000.ssp",
        "budget": 200,
        "candidate": "sample",
        "plan_digest": plan_identity(plan),
    }
    plan.save(tmp_path / "plans" / winner["file"])
    write(tmp_path / "plans/index.json", [winner])
    write(tmp_path / "programs/index.json", {"world_size": 1})
    write(
        tmp_path / "request.json",
        {"factory": "must_not_import:factory", "device": "cuda:0"},
    )
    write(
        tmp_path / "execution/budget-0000/result.json",
        {
            **winner,
            "per_rank": [{"logical_rank": 0, "steps": [{"step": 1}]}],
        },
    )
    return tmp_path


def settings(root):
    return SimpleNamespace(
        quickstart=root,
        nsys="nsys",
        warmup_steps=3,
        capture_steps=3,
        capture_settle_seconds=1,
        gpu_metrics_frequency=10000,
    )


def test_selects_only_completed_winner_and_declared_world(saved_winner):
    root = saved_winner
    _, _, selected = completed(root, budget_gib=200 / (1 << 30))
    assert selected[0]["candidate"] == "sample"
    with pytest.raises(ValueError, match="no completed"):
        completed(root, budget_gib=4)
    with pytest.raises(ValueError, match="world size"):
        completed(root, world_size=2)
    (root / "execution/budget-0000/result.json").unlink()
    with pytest.raises(ValueError, match="no completed"):
        completed(root)


@pytest.mark.parametrize(
    "change", ["result_identity", "missing_rank", "no_steps", "plan"]
)
def test_rejects_stale_or_incomplete_evidence(saved_winner, change):
    root = saved_winner
    path = root / "execution/budget-0000/result.json"
    result = read(path)
    if change == "result_identity":
        result["plan_digest"] = "different"
    elif change == "missing_rank":
        result["per_rank"] = []
    elif change == "no_steps":
        result["per_rank"][0]["steps"] = []
    else:
        plan_path = root / "plans/budget-0000.ssp"
        plan = Plan.load(plan_path)
        replace(plan, communication_info=communication(latency=1)).save(plan_path)
    write(path, result)
    with pytest.raises(ValueError):
        completed(root)


def test_capture_reuses_checked_archives_and_not_changed_outputs(saved_winner):
    root = saved_winner
    request, index, winners = completed(root)
    inputs = fingerprint(root, request, index, winners[0], settings(root))
    case = root / "capture"
    case.mkdir()
    assert not verified_capture(case, inputs)
    report, sqlite = case / "winner.nsys-rep", case / "winner.sqlite"
    report.write_bytes(b"report")
    sqlite.write_bytes(b"sqlite")
    write(
        case / "validation.json",
        {
            "inputs": inputs,
            "report": report.name,
            "sqlite": sqlite.name,
            "report_sha256": digest(report),
            "sqlite_sha256": digest(sqlite),
        },
    )
    assert verified_capture(case, inputs)
    with pytest.raises(ValueError, match="different inputs"):
        verified_capture(case, {**inputs, "rank_map": {"0": 1}})
    report.write_bytes(b"changed")
    with pytest.raises(ValueError, match="capture changed"):
        verified_capture(case, inputs)


def test_capture_command_uses_existing_device_worker(saved_winner):
    root = saved_winner
    argv = command(settings(root), root / "winner", 4, 2, plan_digest="saved")
    assert "--nproc-per-node=2" in argv
    assert "benchmarking.device_worker" in argv
    assert "tools.pressurefit_trace.profile_winners" in argv
    assert "--flush-on-cudaprofilerstop=false" in argv
    assert "--capture-range=cudaProfilerApi" in argv
    assert "--gpu-metrics-devices=cuda-visible" in argv
    assert argv[argv.index("--plan-digest") + 1] == "saved"
    assert all(isinstance(arg, str) for arg in argv)


@pytest.mark.parametrize(
    "help_text,expected",
    [
        ("--trace: cuda, nvtx, osrt", "cuda,nvtx,osrt"),
        ("--trace: cuda, nvtx, osrt, nccl", "cuda,nvtx,osrt,nccl"),
    ],
)
def test_capture_uses_only_advertised_trace_types(monkeypatch, help_text, expected):
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: help_text)
    assert supported_traces("nsys") == expected


def test_selection_and_capture_import_without_preparation_or_torch(saved_winner):
    import os
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'must_not_import'} or any(
            fullname == item or fullname.startswith(item + '.')
            for item in (
                'shadowspill.build', 'shadowspill.execution', 'shadowspill.store'
            )
        ):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from tools.pressurefit_trace.profile_winners.capture import completed
assert len(completed(Path(sys.argv[1]))[2]) == 1
"""
    child = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            str(saved_winner),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize(
    "option,value",
    [
        ("--world-size", "0"),
        ("--capture-steps", "0"),
        ("--budget-gib", "nan"),
        ("--capture-settle-seconds", "-1"),
    ],
)
def test_invalid_capture_settings_fail_before_reading_artifacts(
    tmp_path, option, value
):
    from tools.pressurefit_trace.profile_winners.__main__ import main

    with pytest.raises(SystemExit) as error:
        main(
            [str(tmp_path / "absent"), "--output", str(tmp_path / "out"), option, value]
        )
    assert error.value.code == 2
