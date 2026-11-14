"""Pass only explicit CLI choices to the common saved-run configuration."""

import argparse
from collections.abc import Sequence
from typing import Any


def arguments(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> tuple[str, dict[str, Any]]:
    directory = parser.get_default("run_dir")
    for action in parser._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    values = vars(parser.parse_args(argv))
    if directory and "run_dir" not in values:
        values["run_dir"] = directory
    return values.pop("phase", "all"), values
