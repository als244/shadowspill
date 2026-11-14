"""Import only the requested phase's implementation."""

import argparse
import importlib

from .common import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("build", "profile", "communicate", "plan", "execute")
    )
    parser.add_argument("request")
    parser.add_argument("--selection", type=int)
    args = parser.parse_args()
    phase = importlib.import_module("." + args.phase, __package__)
    request = read(args.request)
    if args.phase == "execute":
        phase.run(request, args.selection)
    else:
        phase.run(request)


if __name__ == "__main__":
    main()
