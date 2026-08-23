from __future__ import annotations

from pathlib import Path

import pytest

from shadowspill._libraries import (
    LIBRARY_DIRECTORY_ENVIRONMENT,
    library_candidates,
    shadowspill_library_path,
)
from shadowspill.pytorch.runtime_adapter.runtime import _adapter_path


def _editable_checkout(tmp_path: Path) -> Path:
    package = tmp_path / "project" / "src" / "shadowspill"
    package.mkdir(parents=True)
    (tmp_path / "project" / "pyproject.toml").touch()
    (tmp_path / "project" / "CMakeLists.txt").touch()
    return package


def test_packaged_and_editable_candidates_have_stable_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LIBRARY_DIRECTORY_ENVIRONMENT, raising=False)
    package = _editable_checkout(tmp_path)

    candidates = library_candidates(
        "libshadowspill_example.so",
        package_root=package,
    )

    assert candidates[0] == package / "lib" / "libshadowspill_example.so"
    assert candidates[1].parent.name.startswith("cp")
    assert len(candidates) == 2


def test_no_build_directory_is_searched_unless_it_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale library somewhere the loader merely knows about is silently wrong."""

    monkeypatch.delenv(LIBRARY_DIRECTORY_ENVIRONMENT, raising=False)
    package = _editable_checkout(tmp_path)
    stale = tmp_path / "project" / "build" / "dev"

    candidates = library_candidates(
        "libshadowspill_example.so",
        package_root=package,
    )

    assert stale not in {candidate.parent for candidate in candidates}


def test_a_named_directory_is_searched_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _editable_checkout(tmp_path)
    named = tmp_path / "elsewhere"
    monkeypatch.setenv(LIBRARY_DIRECTORY_ENVIRONMENT, str(named))

    candidates = library_candidates(
        "libshadowspill_example.so",
        package_root=package,
    )

    assert candidates[0] == named / "libshadowspill_example.so"
    assert len(candidates) == 3


def test_editable_build_libraries_are_discovered_without_environment() -> None:
    library = shadowspill_library_path()
    if library is None:
        pytest.skip("the editable library has not been built")

    assert library.name.startswith("libshadowspill.so")
    try:
        adapter = _adapter_path(None)
    except RuntimeError:
        pytest.skip("the optional PyTorch adapter has not been built")
    assert adapter.name == "libshadowspill_pytorch.so"
