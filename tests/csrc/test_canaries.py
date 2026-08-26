"""The C canaries, run from the suite everyone runs.

The canaries are CTest tests, and CMake stays the authority on which ones
exist, what they are labelled, and how long they may take. This runs them
through `ctest` rather than listing them again here, so a canary added to
`CMakeLists.txt` is picked up without touching this file.

They were previously reachable only by running `ctest` by hand, which nothing
in the repository did -- so a canary could fail for a long time without
anyone noticing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

#: `build/{wheel_tag}`, as `pyproject.toml` configures scikit-build-core.
_BUILD_ROOT = Path(__file__).resolve().parents[2] / "build"


def _build_directory() -> Path | None:
    """The CMake build tree, or None if there is not one.

    An installed wheel carries the library beside the package and has no
    build tree, so there is nothing to run and nothing to report.
    """

    for testfile in sorted(_BUILD_ROOT.glob("*/CTestTestfile.cmake")):
        return testfile.parent
    return None


def _run_ctest(*selectors: str) -> None:
    build = _build_directory()
    if build is None:
        pytest.skip("no CMake build tree; canaries are built by an editable install")
    if shutil.which("ctest") is None:
        pytest.skip("ctest is not on PATH")
    completed = subprocess.run(
        ["ctest", "--output-on-failure", "--no-tests=error", *selectors],
        cwd=build,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        # The canaries report by exit status and print little, so the whole
        # output is the evidence.
        pytest.fail(
            f"ctest {' '.join(selectors)} failed with {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def test_c_canaries_pass() -> None:
    """Every canary that does not need an accelerator."""

    _run_ctest("--label-exclude", "cuda")


@pytest.mark.cuda
def test_cuda_c_canaries_pass() -> None:
    """The canaries CMake labelled `cuda`, which need the qualified backend."""

    _run_ctest("--label-regex", "cuda")
