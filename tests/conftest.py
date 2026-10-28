"""Suite options and source roots inherited by every process the tests start."""

import os

import pytest

from tests.precision import select_test_dtype


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--test-dtype",
        choices=("float16", "bfloat16"),
        default=None,
        help="low-precision test dtype (default: BF16, FP16 below SM80)",
    )


def pytest_configure(config: pytest.Config) -> None:
    try:
        selected = select_test_dtype(config.getoption("--test-dtype"))
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    # Resolve before test collection and inherit into all CTest subprocesses.
    os.environ["SHADOWSPILL_TEST_DTYPE"] = selected
    # pyproject's `pythonpath` puts the source roots on this process's path
    # only. A worker the tests launch by file path has its own directory first
    # and the repository root nowhere, so it could not import `tests.*` or the
    # top-level packages; the roots travel to it through the environment.
    roots = [str(path) for path in config.getini("pythonpath")]
    inherited = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*(root for root in roots if root not in inherited), *inherited]
    )
