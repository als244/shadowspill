from __future__ import annotations

from pathlib import Path

import pytest

from shadowspill._libraries import library_candidates
from shadowspill.planner._capi import planner_library_path
from shadowspill.pytorch.runtime_adapter.runtime import _adapter_path
from shadowspill.simulator._capi import simulator_library_path


def test_packaged_and_editable_candidates_have_stable_precedence(
    tmp_path: Path,
) -> None:
    package = tmp_path / "project" / "src" / "shadowspill"
    package.mkdir(parents=True)
    (tmp_path / "project" / "pyproject.toml").touch()
    (tmp_path / "project" / "CMakeLists.txt").touch()

    candidates = library_candidates(
        "libshadowspill_example.so",
        package_root=package,
    )

    assert candidates[0] == package / "lib" / "libshadowspill_example.so"
    assert candidates[1].parent.name.startswith("cp")
    assert candidates[2] == (
        tmp_path / "project" / "build" / "dev" / "libshadowspill_example.so"
    )
    assert len(candidates) == 3


def test_editable_build_libraries_are_discovered_without_environment() -> None:
    planner = planner_library_path()
    simulator = simulator_library_path()
    if planner is None or simulator is None:
        pytest.skip("compiled editable libraries have not been built")

    # Planning, simulation and execution ship as one library, so both
    # lookups resolve to it.
    assert planner.name.startswith("libshadowspill.so")
    assert simulator.name.startswith("libshadowspill.so")
    assert planner == simulator
    try:
        adapter = _adapter_path(None)
    except RuntimeError:
        pytest.skip("the optional PyTorch adapter has not been built")
    assert adapter.name == "libshadowspill_pytorch.so"
