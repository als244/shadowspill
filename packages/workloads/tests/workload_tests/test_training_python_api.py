"""Python model descriptions use the same saved runner without importing CUDA."""

import json
import subprocess
import sys
from fractions import Fraction

import pytest

from shadowspill import ProfilingOptions
from shadowspill.training.preparation import api as runner
from shadowspill_workloads.recipes import LoRA, Model, train


@pytest.mark.parametrize(
    "name",
    [
        "llama3-1b",
        "olmoe-7b-a1b",
        "qwen3-1.7b",
        "qwen3moe-30b-a3b",
        "qwen35-2b",
        "qwen35moe-35b-a3b",
    ],
)
def test_catalog_training_uses_common_defaults_and_freezes_search(
    tmp_path, monkeypatch, name
):
    launches = []
    monkeypatch.setattr(runner, "run_phases", lambda *a, **k: launches.append((a, k)))
    model = Model.from_config(name)
    root = train(
        model,
        dict(
            run_dir=tmp_path / "run",
            data=tmp_path / "data",
            steps=100,
            sequences_per_step=4,
            microbatch_sizes=[1, 2, 4],
        ),
        phase="build",
    )
    saved = json.loads((root / "config.json").read_text())
    assert saved["model"] == name
    assert saved["schedule_steps"] == 100
    assert saved["sequence_length"] == 2048
    assert saved["initialization"] == "random"
    assert launches[0][1] == {"ranks": 1, "preparation_ranks": 1}
    assert [c["num_examples"] for c in saved["candidates"]] == [4, 2, 1]
    train(model, {"run_dir": root, "execution_gib": 20}, phase="plan")
    restored = json.loads((root / "config.json").read_text())
    assert restored["candidates"] == saved["candidates"]
    assert restored["model"] == name
    with pytest.raises(ValueError, match="schedule_steps"):
        train(model, {"run_dir": root, "steps": 101}, phase="execute")
    assert len(launches) == 2


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("llama3-tiny", {"parallelism": "ep", "ranks": 2}, "expert parallelism"),
        ("olmoe-tiny", {"parallelism": "ep", "ranks": 3}, "divisible by ranks"),
        ("qwen3-tiny", {"learning_rate": float("nan")}, "finite and positive"),
    ],
)
def test_catalog_rejects_invalid_training_before_launch(
    tmp_path, monkeypatch, name, options, message
):
    launches = []
    monkeypatch.setattr(runner, "run_phases", lambda *a, **k: launches.append((a, k)))
    with pytest.raises(ValueError, match=message):
        train(
            Model.from_config(name),
            dict(run_dir=tmp_path / "run", data=tmp_path / "data", **options),
        )
    assert not launches and not (tmp_path / "run").exists()


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
            tail_save_counts=[4, 0, 1, 4],
            profiling=ProfilingOptions(conditioning_max_samples=7, maximum_samples=11),
        ),
        phase="build",
    )
    saved = json.loads((root / "config.json").read_text())
    assert len(saved["candidates"]) == 12
    assert saved["tail_save_counts"] == [0, 1, 4]
    assert saved["profiling"]["conditioning_max_samples"] == 7
    assert saved["profiling"]["maximum_samples"] == 11
    from shadowspill.training.preparation.search import search_options

    planner = search_options(saved).resolved_algorithm
    assert planner.options.tail_save_counts == (0, 1, 4)
    assert planner.options.resolution_options == tuple(Fraction(n, 4) for n in range(5))
    assert {c["num_examples"] for c in saved["candidates"]} == {2, 4}
    train(model, {"run_dir": root}, phase="build")
    assert json.loads((root / "config.json").read_text()) == saved
    with pytest.raises(ValueError, match="fresh run"):
        train(model, {"run_dir": root}, phase="all")
    with pytest.raises(ValueError, match="fresh run"):
        train(model, {"run_dir": root, "chunk_sizes": [4]}, phase="build")
    train(model, {"run_dir": root, "execution_gib": 24}, phase="plan")
    assert [call[0][0] for call in launches] == ["shadowspill.training.preparation"] * 3
    assert (
        json.loads((root / "config.json").read_text())["candidates"]
        == saved["candidates"]
    )
    with pytest.raises(ValueError, match="fresh run"):
        train(model, {"run_dir": root, "chunk_sizes": [4]}, phase="plan")
    train(model, {"run_dir": root, "refresh_profiles": True}, phase="profile")
    train(
        model, {"run_dir": root, "profiling": {"maximum_samples": 13}}, phase="profile"
    )
    assert (
        json.loads((root / "config.json").read_text())["profiling"]["maximum_samples"]
        == 13
    )
    with pytest.raises(ValueError, match="phase='profile'"):
        train(
            model,
            {"run_dir": root, "profiling": {"maximum_samples": 14}},
            phase="execute",
        )
    with pytest.raises(ValueError, match="phase='profile'"):
        train(model, {"run_dir": root, "refresh_profiles": False}, phase="execute")
    with pytest.raises(ValueError, match="boolean"):
        train(model, {"run_dir": root, "refresh_profiles": "yes"}, phase="profile")
    train(model, {"run_dir": root, "tail_save_counts": [0, 3]}, phase="plan")
    assert (
        json.loads((root / "config.json").read_text())["candidates"]
        == saved["candidates"]
    )
    with pytest.raises(ValueError, match="phase='plan'"):
        train(model, {"run_dir": root, "tail_save_counts": [0, 2]}, phase="execute")
    # Capture is a shared execution control, including for this non-GLM recipe.
    train(
        model, {"run_dir": root, "nsys_start_step": 3, "nsys_steps": 2}, phase="execute"
    )
    captured = json.loads((root / "config.json").read_text())
    assert captured["nsys_start_step"] == 3 and captured["nsys_steps"] == 2
    assert captured["candidates"] == saved["candidates"]
    with pytest.raises(ValueError, match="cadence"):
        train(model, {"run_dir": root, "nsys_start_step": 2}, phase="execute")


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
            sequences_per_step=64,
            microbatch_sizes=[4, 16],
        ),
        phase="build",
    )
    saved = json.loads((root / "config.json").read_text())
    assert launches[0][1] == {"ranks": 8, "preparation_ranks": 8}
    assert [c["num_examples"] for c in saved["candidates"]] == [16, 4]
    assert saved["initialization"] == "random"


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "glm5_next"])
def test_benchmark_configuration_uses_random_host_values_without_checkpoints(
    tmp_path, monkeypatch, model_type
):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    model = Model.from_checkpoint(directory, lora=LoRA())
    monkeypatch.setattr(runner, "run_phases", lambda *a, **k: None)
    root = train(
        model,
        dict(
            run_dir=tmp_path / "benchmark",
            data=tmp_path / "data",
            initialization="random",
            seed=17,
            checkpoint_every=0,
            eval_every=0,
        ),
        phase="build",
    )
    options = json.loads((root / "config.json").read_text())
    assert options["initialization"] == "random"
    assert options["checkpoint_every"] == options["eval_every"] == 0
    assert options["frozen_state_id"] is None
    from shadowspill_workloads.recipes.text.training import frozen_state_identity

    first = frozen_state_identity(options, {"model_type": model_type})
    assert first.startswith("random-v1:")
    assert first != frozen_state_identity(
        {**options, "seed": 18}, {"model_type": model_type}
    )
    with pytest.raises(ValueError, match="frozen_state_id"):
        train(
            model,
            {"run_dir": tmp_path / "real", "data": tmp_path / "data"},
            phase="build",
        )


def test_budget_runs_reuse_completed_measurements_without_mutating_source(
    tmp_path, monkeypatch
):
    from shadowspill.training.preparation.candidates import program_paths

    monkeypatch.setattr(runner, "run_phases", lambda *a, **k: None)
    model = Model.from_config("qwen3moe-tiny")
    parent = train(
        model,
        dict(
            run_dir=tmp_path / "prepared",
            data=tmp_path / "data",
            ranks=1,
            sequences_per_step=64,
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
        dict(
            run_dir=tmp_path / "budget20",
            prepared_from=parent,
            execution_gib=20,
            checkpoint_every=0,
        ),
        phase="plan",
    )
    selected = json.loads((child / "config.json").read_text())
    assert selected["candidates"] == options["candidates"]
    assert selected["resume"] is None
    assert selected["execution_gib"] == 20
    assert selected["checkpoint_every"] == 0
    assert (child / "profiled").is_symlink()
    assert (parent / "config.json").read_bytes() == source_config
    for phase in ("build", "profile", "communicate"):
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
from shadowspill_workloads.recipes import Model, LoRA, train
for name in ('llama3', 'olmoe', 'qwen3', 'qwen35', 'qwen3moe', 'qwen35moe'):
    assert Model.from_config(f'{name}-tiny')
runpy.run_module('examples.train_text')
for name in ('qwen35_lora', 'glm53_flash_lora', 'qwen3'):
    runpy.run_module(f'examples.{name}')
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

    from shadowspill.training.preparation.candidates import (
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
