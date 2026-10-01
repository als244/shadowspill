"""Run the supplied packed-text recipe from a JSON configuration."""

from __future__ import annotations

import sys

from workloads.recipes.text.run import run_config


def main(arguments: list[str]) -> None:
    if not arguments:
        raise SystemExit(__doc__)
    path, *overrides = arguments
    run_config(path, overrides)


if __name__ == "__main__":
    main(sys.argv[1:])
