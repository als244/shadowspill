"""Command-line paths and connection settings shared by the demo files."""

import argparse
from pathlib import Path


def arguments(name, dataset, *, saved_plan=False):
    parser = argparse.ArgumentParser(description=name.replace("_", " "))
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument(
        "--host", default="training-host", help="OpenSSH host alias"
    )
    connection.add_argument(
        "--url", help="HTTPS service URL or an existing loopback tunnel"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="service port on that host"
    )
    parser.add_argument("--data", type=Path, default=Path("data") / dataset)
    parser.add_argument("--output", type=Path, default=Path("runs/demos") / name)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--repeat-first-batch",
        action="store_true",
        help="repeat real data for a short learning check",
    )
    if saved_plan:
        parser.add_argument(
            "--prepared",
            type=Path,
            required=True,
            help="directory with plans.ssp and training-config.json",
        )
    options = parser.parse_args()
    if options.steps < 1:
        parser.error("--steps must be positive")
    options.output.mkdir(parents=True, exist_ok=True)
    return options
