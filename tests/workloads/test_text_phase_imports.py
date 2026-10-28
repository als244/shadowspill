"""Recipe phase modules retain their CPU/GPU and preparation boundaries."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("recipe", ["llama3", "qwen35_lora", "glm53_flash_lora"])
@pytest.mark.parametrize("phase", ["plan", "execute"])
def test_recipe_import_boundary(recipe, phase):
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            """
import importlib
import sys
recipe, phase = sys.argv[1:]
if phase == "execute":
    from tests.shadowspill.execution._cold import forbid_preparation
    forbid_preparation()
importlib.import_module(f"workloads.recipes.text.{recipe}.{phase}")
if phase == "plan":
    assert "torch" not in sys.modules
    assert "workloads.models" not in sys.modules
""",
            recipe,
            phase,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
