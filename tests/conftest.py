"""Suite options inherited by fresh-process CTest canaries."""

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--test-dtype",
        choices=("float16", "bfloat16"),
        default=None,
        help="dtype for generic low-precision GPU tests (default: bfloat16)",
    )


def pytest_configure(config: pytest.Config) -> None:
    selected = config.getoption("--test-dtype")
    if selected is not None:
        os.environ["SHADOWSPILL_TEST_DTYPE"] = selected
    if os.environ.get("SHADOWSPILL_TEST_DTYPE", "bfloat16") not in {
        "float16",
        "bfloat16",
    }:
        raise pytest.UsageError("SHADOWSPILL_TEST_DTYPE must be float16 or bfloat16")
