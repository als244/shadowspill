"""Python model descriptions use the same saved runner without importing CUDA."""

import json
import subprocess
import sys

import pytest

from workloads.recipes import LoRA, Model, train
from workloads.recipes.training import api as runner


def test_checkpoint_model_and_method_select_frozen_candidates(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    model = Model.from_checkpoint(checkpoint, lora=LoRA(rank=32, alpha=32))
    launches = []
    monkeypatch.setattr(runner, "run_phases", lambda *a, **k: launches.append((a, k)))
    root = train(
        model,
        dict(
            run_dir=tmp_path / "run",
            data=tmp_path / "data",
            frozen_state_id="fixture",
            sequence_length=17,
            sequences_per_step=4,
            microbatch_sizes=[1, 2],
            chunk_sizes=[8],
            moe_chunk_sizes=[8, 16],
            graph_pair_policies=["cost-input", "cost-attention", "attention-input"],
        ),
        phase="build",
    )
    saved = json.loads((root / "config.json").read_text())
    assert len(saved["candidates"]) == 12
    assert {c["num_examples"] for c in saved["candidates"]} == {2, 4}
    train(model, {"run_dir": root, "execution_gib": 24}, phase="plan")
    assert [call[0][0] for call in launches] == ["workloads.recipes.training"] * 2
    assert (
        json.loads((root / "config.json").read_text())["candidates"]
        == saved["candidates"]
    )
    with pytest.raises(ValueError, match="fresh run"):
        train(model, {"run_dir": root, "chunk_sizes": [4]}, phase="plan")


def test_ep_geometry_declares_collective_preparation_before_launch(
    tmp_path, monkeypatch
):
    launches = []
    monkeypatch.setattr(runner, "run_phases", lambda *a, **k: launches.append((a, k)))
    root = train(
        Model.from_config("qwen3moe-30b-a3b"),
        dict(
            run_dir=tmp_path / "run",
            data=tmp_path / "data",
            ranks=8,
            parallelism="ep",
            microbatch_sizes=[4, 16],
        ),
        phase="build",
    )
    saved = json.loads((root / "config.json").read_text())
    assert launches[0][1] == {"ranks": 8, "preparation_ranks": 8}
    assert [c["num_examples"] for c in saved["candidates"]] == [16, 4]


def test_budget_runs_reuse_completed_measurements_without_mutating_source(
    tmp_path, monkeypatch
):
    from workloads.recipes.training.candidates import program_paths

    monkeypatch.setattr(runner, "run_phases", lambda *a, **k: None)
    model = Model.from_config("qwen3moe-tiny")
    parent = train(
        model,
        dict(
            run_dir=tmp_path / "prepared",
            data=tmp_path / "data",
            ranks=1,
            microbatch_sizes=[4, 16],
            resume=tmp_path / "old-checkpoint.ssp",
        ),
        phase="build",
    )
    source_config = (parent / "config.json").read_bytes()
    options = json.loads(source_config)
    for relative in program_paths(options):
        for name in ("programs", "profiled", "communication"):
            path = parent / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"prepared artifact")
    for candidate in options["candidates"]:
        (parent / "programs" / candidate["path"] / "metadata.json").write_text("{}")
    child = train(
        model,
        dict(run_dir=tmp_path / "budget20", prepared_from=parent, execution_gib=20),
        phase="plan",
    )
    selected = json.loads((child / "config.json").read_text())
    assert selected["candidates"] == options["candidates"]
    assert selected["resume"] is None
    assert selected["execution_gib"] == 20
    assert (child / "profiled").is_symlink()
    assert (parent / "config.json").read_bytes() == source_config
    for phase in ("profile", "communicate"):
        with pytest.raises(ValueError, match="only plan or execute"):
            train(model, {"run_dir": child}, phase=phase)
    with pytest.raises(ValueError, match="fresh run"):
        train(model, {"run_dir": child, "microbatch_sizes": [8]}, phase="plan")
    for destination in (parent / "profiled/child", parent / "plans/child"):
        with pytest.raises(ValueError, match="outside prepared_from"):
            train(
                model,
                {"run_dir": destination, "prepared_from": parent},
                phase="plan",
            )
    (parent / "communication" / program_paths(options)[0]).unlink()
    with pytest.raises(ValueError, match="completed measurement"):
        train(
            model,
            {"run_dir": tmp_path / "incomplete", "prepared_from": parent},
            phase="plan",
        )
    assert not (tmp_path / "incomplete").exists()


def test_public_imports_and_examples_do_not_import_frameworks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, runpy
sys.modules['torch'] = None
sys.modules['mlops'] = None
from workloads.recipes import Model, LoRA, train
assert Model.from_config('qwen3moe-tiny')
for name in ('qwen35_lora', 'glm53_flash_lora', 'qwen3'):
    runpy.run_module(f'workloads.recipes.text.{name}.example')
assert sys.modules['torch'] is None and sys.modules['mlops'] is None
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_candidate_exhaustion_is_recorded_but_contract_errors_stop(tmp_path):
    from types import SimpleNamespace

    import torch

    from workloads.recipes.training.candidates import (
        attempt_candidate,
        available,
        failures,
        program_paths,
    )

    options = {
        "candidates": [{"path": "large"}, {"path": "small"}],
        "eval_every": 1,
    }

    def exhausted():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    attempt_candidate(
        tmp_path,
        options["candidates"][0],
        "profile",
        exhausted,
        SimpleNamespace(count=1),
    )
    assert failures(tmp_path)["large"]["phase"] == "profile"
    assert available(tmp_path, options) == ({"path": "small"},)
    assert program_paths(options, root=tmp_path) == (
        "small/training.ssp",
        "small/evaluation.ssp",
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        attempt_candidate(
            tmp_path,
            {"path": "collective"},
            "build",
            exhausted,
            SimpleNamespace(count=2),
        )

    def invalid():
        raise ValueError("invalid execution-axis contract")

    with pytest.raises(ValueError, match="execution-axis"):
        attempt_candidate(
            tmp_path, {"path": "invalid"}, "build", invalid, SimpleNamespace(count=1)
        )
    assert set(failures(tmp_path)) == {"large"}
