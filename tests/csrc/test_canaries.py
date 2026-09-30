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
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
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


def _failed_canaries(output: str) -> tuple[str, ...]:
    """The canaries ctest named in its closing summary.

    Without this a failure is reported as the wrapper that ran ctest, and
    which canary broke is only findable by reading the whole dump.
    """

    lines = output.splitlines()
    try:
        start = lines.index("The following tests FAILED:") + 1
    except ValueError:
        return ()
    named = []
    for line in lines[start:]:
        match = re.match(r"\s*\d+ - (\S+) \(", line)
        if match is None:
            break
        named.append(match.group(1))
    return tuple(named)


def _stream_process(
    command: list[str], *, cwd: Path, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Show progress and enforce a deadline on the whole child process group."""

    output: list[bytes] = []
    started = time.monotonic()
    deadline = started + timeout
    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        assert process.stdout is not None
        try:
            with selectors.DefaultSelector() as reader:
                reader.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    if not reader.select(timeout=min(30, remaining)):
                        print(
                            f"\n{command[0]} still running after "
                            f"{time.monotonic() - started:.0f}s",
                            flush=True,
                        )
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    output.append(chunk)
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
        except BaseException:
            # Killing only ctest leaves its Python/CUDA children behind.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return subprocess.CompletedProcess(
        command, returncode, stdout=b"".join(output).decode("utf-8", errors="replace")
    )


def _run_ctest(*selectors: str) -> None:
    build = _build_directory()
    if build is None:
        pytest.skip("no CMake build tree; canaries are built by an editable install")
    if shutil.which("ctest") is None:
        pytest.skip("ctest is not on PATH")
    try:
        completed = _stream_process(
            ["ctest", "--output-on-failure", "--no-tests=error", *selectors],
            cwd=build,
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
        failed = _failed_canaries(completed.stdout)
        # Lead with the canary, because that first line is what a summary
        # shows. The canaries report by exit status and print little, so the
        # whole output follows as the evidence.
        headline = (
            ", ".join(failed)
            if failed
            else f"ctest {' '.join(selectors)} (no canary named)"
        )
        pytest.fail(
            f"{headline} failed with {completed.returncode}\n{completed.stdout}"
        )


def test_c_canaries_pass(capsys: pytest.CaptureFixture[str]) -> None:
    """Every canary that needs neither an accelerator nor a peer.

    ``--label-exclude`` takes a regular expression, not a list: the two labels
    are joined with ``|``, and writing them with a semicolon would exclude
    nothing and silently run both sets here.
    """

    with capsys.disabled():
        _run_ctest("--label-exclude", "cuda|network")


@pytest.mark.cuda
def test_cuda_c_canaries_pass(capsys: pytest.CaptureFixture[str]) -> None:
    """The canaries CMake labelled `cuda`, which need the qualified backend."""

    with capsys.disabled():
        _run_ctest("--label-regex", "cuda")


@pytest.mark.network
def test_network_c_canaries_pass(capsys: pytest.CaptureFixture[str]) -> None:
    """The canaries CMake labelled `network`, which need a memory daemon.

    No peer means this optional gate skips, including builds where the network
    extension and its CTest targets were not built.
    """

    if not os.environ.get("SHADOWSPILL_NETWORK_PEER"):
        pytest.skip("no network memory peer configured")
    with capsys.disabled():
        _run_ctest("--label-regex", "network")
