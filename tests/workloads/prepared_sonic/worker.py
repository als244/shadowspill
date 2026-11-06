"""One phase per process; planning and the independent oracle are CPU-only."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "build",
            "profile",
            "plan",
            "execute",
            "resume",
            "inspect",
        ),
    )
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    directory = arguments.directory.resolve()
    case = json.loads((directory / "case.json").read_text())
    if case not in (
        {"batch": 2, "variant": "recompute", "kind": "forward_backward"},
        {"batch": 2, "variant": "save", "kind": "step"},
    ):
        raise ValueError("unsupported prepared-Sonic integration case")
    from . import phases

    if arguments.phase == "resume":
        phases.execute(directory, case, resume=True)
    else:
        getattr(phases, arguments.phase)(directory, case)


if __name__ == "__main__":
    main()
