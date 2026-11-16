"""Recipe phase modules retain their CPU/GPU and preparation boundaries."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "package",
    ["shadowspill.training.preparation"],
    ids=["shared"],
)
@pytest.mark.parametrize("phase", ["plan", "execute"])
def test_recipe_import_boundary(package, phase):
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            """
import importlib
import sys
package, phase = sys.argv[1:]
if phase == "execute":
    from engine_tests.execution._cold import forbid_preparation
    forbid_preparation()
importlib.import_module(f"{package}.{phase}")
if phase == "plan":
    assert "torch" not in sys.modules
    assert "shadowspill_workloads.models" not in sys.modules
""",
            package,
            phase,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
