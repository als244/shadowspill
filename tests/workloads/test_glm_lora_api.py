"""Python recipe configuration preserves the saved CLI/worker contracts."""

import json
import subprocess
import sys

import pytest

from workloads.recipes.text.glm53_flash_lora import api


def test_python_configuration_preserves_candidates_and_saved_phase_overrides(
    tmp_path, monkeypatch
):
    launches = []
    monkeypatch.setattr(
        api, "run_phases", lambda *args, **kw: launches.append((args, kw))
    )
    config = {
        "run_dir": tmp_path / "run",
        "checkpoint": tmp_path / "checkpoint",
        "data": tmp_path / "data",
        "frozen_state_id": "fixture",
        "ranks": 8,
        "chunk_sizes": [0, 4096, 8192],
        "graph_pair_policies": ["cost-input", "cost-attention", "attention-input"],
        "execution_gib": 48,
        "spill_gib": 96,
    }
    root = api.run(config, phase="build")
    saved = json.loads((root / "config.json").read_text())
    assert len(saved["build_variants"]) == 9
    assert saved["lora_rank"] == saved["lora_alpha"] == 32
    assert saved["checkpoint"] == str(config["checkpoint"])
    assert config["run_dir"] == root  # Caller-owned Path remains a Path.
    assert launches[-1] == (
        ("workloads.recipes.text.glm53_flash_lora", root, "build"),
        {"ranks": 8},
    )

    api.run({"run_dir": root, "execution_gib": 40}, phase="plan")
    assert json.loads((root / "config.json").read_text()) == {
        **saved,
        "execution_gib": 40,
    }
    api.run({"run_dir": root, "resume": tmp_path / "step.ssp"}, phase="execute")
    resumed = json.loads((root / "config.json").read_text())
    assert resumed["resume"] == str(tmp_path / "step.ssp")
    assert resumed["build_variants"] == saved["build_variants"]
    assert launches[-1][0][-1] == "execute" and launches[-1][1] == {"ranks": 8}
    with pytest.raises(ValueError, match="changed budgets"):
        api.run({"run_dir": root, "execution_gib": 48}, phase="execute")
    assert json.loads((root / "config.json").read_text()) == resumed


def test_unknown_options_and_invalid_phases_fail_before_creating_a_run(tmp_path):
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="unknown recipe options: chunk_size"):
        api.run({"run_dir": root, "chunk_size": 8192})
    with pytest.raises(ValueError, match="unknown recipe phase"):
        api.run({"run_dir": root}, phase="planning")
    with pytest.raises(ValueError, match="pass phase"):
        api.run({"run_dir": root, "phase": "build"})
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
from workloads.recipes.text.glm53_flash_lora import api
def refuse(*args, **kwargs):
    raise AssertionError('import started a worker')
api.run_phases = refuse
example = runpy.run_module('workloads.recipes.text.glm53_flash_lora.example')
assert example['config']['chunk_sizes'][0] == 0
assert example['config']['spill_gib'] == 96
assert sys.modules['torch'] is None and sys.modules['mlops'] is None
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
