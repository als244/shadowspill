"""Suite options inherited by fresh-process CTest canaries."""

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
