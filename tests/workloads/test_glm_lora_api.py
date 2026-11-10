"""Python recipe configuration preserves the saved CLI/worker contracts."""

import json
import subprocess
import sys

import pytest

from workloads.recipes.text.glm53_flash_lora import api
from workloads.recipes.training import api as runner
from workloads.recipes.training.candidates import build_candidates


def test_python_configuration_preserves_candidates_and_saved_phase_overrides(
    tmp_path, monkeypatch
):
    launches = []
    monkeypatch.setattr(
        runner, "run_phases", lambda *args, **kw: launches.append((args, kw))
    )
    config = {
        "run_dir": tmp_path / "run",
        "checkpoint": tmp_path / "checkpoint",
        "data": tmp_path / "data",
        "frozen_state_id": "fixture",
        "ranks": 8,
        "sequences_per_step": 8,
        "microbatch_sizes": [2, 4],
        "chunk_sizes": [0, 4096, 8192],
        "graph_pair_policies": ["cost-input", "cost-attention", "attention-input"],
        "execution_gib": 48,
        "spill_gib": 96,
    }
    root = api.run(config, phase="build")
    saved = json.loads((root / "config.json").read_text())
    assert len(saved["candidates"]) == 18
    assert saved["microbatch_sizes"] == [2, 4]
    candidates = build_candidates(saved)
    assert len(candidates) == 18
    assert {c["sequences_per_example"] for c in candidates} == {2, 4}
    assert saved["lora_rank"] == saved["lora_alpha"] == 32
    assert saved["checkpoint"] == str(config["checkpoint"])
    assert config["run_dir"] == root  # Caller-owned Path remains a Path.
    assert launches[-1] == (
        ("workloads.recipes.training", root, "build"),
        {"ranks": 8, "preparation_ranks": 1},
    )

    api.run(
        {
            "run_dir": root,
            "execution_gib": 40,
            "program_workers": 3,
            "search_workers": 0,
        },
        phase="plan",
    )
    assert json.loads((root / "config.json").read_text()) == {
        **saved,
        "execution_gib": 40,
        "program_workers": 3,
        "search_workers": 0,
    }
    api.run({"run_dir": root, "resume": tmp_path / "step.ssp"}, phase="execute")
    resumed = json.loads((root / "config.json").read_text())
    assert resumed["resume"] == str(tmp_path / "step.ssp")
    assert resumed["candidates"] == saved["candidates"]
    assert launches[-1][0][-1] == "execute" and launches[-1][1] == {
        "ranks": 8,
        "preparation_ranks": 1,
    }
    with pytest.raises(ValueError, match="changed budgets"):
        api.run({"run_dir": root, "execution_gib": 48}, phase="execute")
    assert json.loads((root / "config.json").read_text()) == resumed
    with pytest.raises(ValueError, match="changed microbatch_sizes"):
        api.run({"run_dir": root, "microbatch_sizes": [1, 2, 4]}, phase="plan")
    assert json.loads((root / "config.json").read_text()) == resumed
    assert [call[0][-1] for call in launches] == ["build", "plan", "execute"]


def test_unknown_options_and_invalid_phases_fail_before_creating_a_run(tmp_path):
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="unknown recipe options: chunk_size"):
        api.run({"run_dir": root, "chunk_size": 8192})
    with pytest.raises(ValueError, match="unknown recipe phase"):
        api.run({"run_dir": root}, phase="planning")
    with pytest.raises(ValueError, match="pass phase"):
        api.run({"run_dir": root, "phase": "build"})
    with pytest.raises(ValueError, match="unknown recipe options: max_sequences"):
        api.run({"run_dir": root, "max_sequences_per_example": 4})
    assert not root.exists()


def test_invalid_microbatch_geometry_fails_before_creating_or_launching_a_run(
    tmp_path, monkeypatch
):
    def refuse(*args, **kwargs):
        pytest.fail("invalid configuration launched a worker")

    monkeypatch.setattr(runner, "run_phases", refuse)
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="must divide sequences_per_step"):
        api.run(
            dict(
                run_dir=root,
                checkpoint=tmp_path / "checkpoint",
                data=tmp_path / "data",
                frozen_state_id="fixture",
                sequences_per_step=8,
                microbatch_sizes=[3],
            )
        )
    assert not root.exists()


def test_commented_example_imports_without_frameworks_or_starting_workers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy, sys
sys.modules['torch'] = None
sys.modules['mlops'] = None
from workloads.recipes.training import api as runner
def refuse(*args, **kwargs):
    raise AssertionError('import started a worker')
runner.run_phases = refuse
example = runpy.run_module('workloads.recipes.text.glm53_flash_lora.example')
assert example['config']['chunk_sizes'][0] == 0
assert example['config']['spill_gib'] == 96
assert example['config']['microbatch_sizes'] == [1]
assert sys.modules['torch'] is None and sys.modules['mlops'] is None
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
