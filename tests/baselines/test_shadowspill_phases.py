"""Preparation stays reusable, with explicit ordering and reference-state checks."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from baselines.engines.shadowspill import preparation
from baselines.engines.shadowspill.reference import assemble
from baselines.engines.shadowspill.settings import Settings
from benchmarking.quickstart.phases.common import write
from benchmarking.quickstart.phases.search import orderings, search_options
from shadowspill.artifacts.tensors import TensorChunk


def test_replanning_existing_profiles_does_not_launch_gpu_phases(tmp_path, monkeypatch):
    settings = Settings(model="llama3-tiny", tokens_per_step=128, execution_gib=2)
    root = tmp_path / "phases"
    case = {"description": {"geometry": {"world_size": 1}}}
    write(
        root / "request.json", {"baseline_identity": preparation.identity(settings, 1)}
    )
    write(root / "programs/index.json", {})
    write(root / "profiled/index.json", {})
    (root / "communication.ssp").write_bytes(b"saved calibration")
    calls = []
    monkeypatch.setattr(
        preparation,
        "run_request",
        lambda request, phase: calls.append((request, phase)),
    )
    _, result = preparation.prepare(
        case, replace(settings, execution_gib=4), tmp_path / "case.json", tmp_path
    )
    assert result["reused_computation"]
    assert [phase for _, phase in calls] == ["plan"]
    assert calls[0][0]["search_budgets"] == [4 << 30]
    write(root / "request.json", calls[0][0])
    write(root / "plans/index.json", [])
    calls.clear()
    _, reused = preparation.prepare(
        case, replace(settings, execution_gib=4), tmp_path / "case.json", tmp_path
    )
    assert reused["reused_planning"] and not calls
    # Changed execution shape must not reuse the saved profiles.
    with pytest.raises(ValueError, match="different baseline"):
        preparation.prepare(
            case,
            replace(settings, tokens_per_step=256),
            tmp_path / "case.json",
            tmp_path,
        )


def test_controller_is_not_launched_once_per_gpu():
    from baselines.common.launch import trainer_command

    command = trainer_command("shadowspill", "python", 8, Path("/run"))
    assert "torch.distributed.run" not in command
    assert "baselines.engines.shadowspill.train" in command
    assert "torch.distributed.run" in trainer_command(
        "fsdp2", "python", 8, Path("/run")
    )


def test_baseline_planning_controls_use_existing_planner_policies():
    assert len(orderings(8)) == 4
    assert [(value.depth, value.breadth) for value in orderings(8, "depth-first")] == [
        (8, 1)
    ]
    with pytest.raises(ValueError, match="ordering"):
        orderings(8, "invented")
    options = search_options({"search_workers": 2, "resolution_options": "0,1"})
    assert options.workers == 2
    assert tuple(map(float, options.algorithm.options.resolution_options)) == (0, 1)


def test_reference_exports_reconstruct_shards_and_reject_disagreements():
    request = TensorChunk("weight", (2, 3), torch.float32, 0, 6, (3, 1))
    chunks = [
        (request, torch.arange(3.0)),
        (replace(request, offset=3), torch.arange(3.0, 6.0)),
    ]
    actual = assemble(chunks + chunks)
    assert torch.equal(actual["weight"], torch.arange(6.0).view(2, 3))
    with pytest.raises(ValueError, match="incomplete"):
        assemble(chunks[:1])
    with pytest.raises(ValueError, match="disagree"):
        assemble([*chunks, (request, torch.ones(3))])


def test_catalog_initializer_accepts_wrapped_parameter_chunks():
    from baselines.engines.shadowspill.initialize import Initialize

    request = TensorChunk(
        "model.embed.weight", (512, 128), torch.bfloat16, 0, 512 * 128, (128, 1)
    )
    with Initialize(Settings(model="llama3-tiny")) as fill:
        whole, parts = torch.empty(512 * 128, dtype=torch.bfloat16), []
        fill(request, whole)
        for offset in range(0, whole.numel(), 8192):
            chunk = torch.empty(8192, dtype=whole.dtype)
            fill(replace(request, offset=offset), chunk)
            parts.append(chunk)
    assert torch.equal(whole, torch.cat(parts))
