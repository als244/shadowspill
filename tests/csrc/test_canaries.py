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

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

#: `build/{wheel_tag}`, as `pyproject.toml` configures scikit-build-core.
_BUILD_ROOT = Path(__file__).resolve().parents[2] / "build"

#: A ceiling on the whole ctest invocation, not on any one canary --
#: each of those carries its own TIMEOUT property in CMakeLists.txt.
_CTEST_TIMEOUT_SECONDS = 1800


def _build_directory() -> Path | None:
    """The CMake build tree this interpreter's library was built in.

    An installed wheel carries the library beside the package and has no
    build tree, so there is nothing to run and nothing to report. Several
    build trees can exist side by side -- one per interpreter -- so prefer
    the one whose tag matches the interpreter running the tests rather than
    whichever sorts first.
    """

    trees = sorted(p.parent for p in _BUILD_ROOT.glob("*/CTestTestfile.cmake"))
    if not trees:
        return None
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for tree in trees:
        if tree.name.startswith(tag):
            return tree
    return trees[0]


def _busy_accelerator() -> str | None:
    """What else is holding the accelerator, if anything.

    The canaries each have their own CTest timeout, and those budgets assume
    the device is theirs: measured here the labelled set takes ~104s idle and
    ~218s with something else resident.

    That 2x does not by itself explain the one timeout seen -- a 7s canary
    hitting a 120s limit needs far more than 2x, and what hung it is not
    known. Sharing the device is still the wrong thing to be doing while
    timing canaries, and a run that starts contended cannot say whether a
    later failure is a real regression, so refuse to start rather than
    report a result that cannot be trusted.
    """

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    others = [
        line for line in result.stdout.splitlines()
        if line.strip() and not line.startswith(f"{os.getpid()},")
    ]
    return "; ".join(others) if others else None


def _run_ctest(*selectors: str, needs_accelerator: bool = False) -> None:
    build = _build_directory()
    if build is None:
        pytest.skip("no CMake build tree; canaries are built by an editable install")
    if shutil.which("ctest") is None:
        pytest.skip("ctest is not on PATH")
    if needs_accelerator and (busy := _busy_accelerator()) is not None:
        pytest.skip(f"another process is holding the accelerator: {busy}")
    try:
        completed = subprocess.run(
            ["ctest", "--output-on-failure", "--no-tests=error", *selectors],
            cwd=build,
            capture_output=True,
            text=True,
            check=False,
            # Every canary already has its own CTest timeout; this only stops
            # a wedged ctest from hanging the suite indefinitely.
            timeout=_CTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"ctest {' '.join(selectors)} did not finish within "
            f"{_CTEST_TIMEOUT_SECONDS}s; per-canary timeouts should have "
            "fired first, so ctest itself is wedged"
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

    _run_ctest("--label-regex", "cuda", needs_accelerator=True)
