"""Small convenience entry point; the Python API accepts the same controls."""

import argparse
import json

from .recipe import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", help="existing public training run with a selected Plan"
    )
    parser.add_argument(
        "--output-dir", required=True, help="new benchmark result directory"
    )
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--units-per-step", type=float)
    parser.add_argument("--unit-label", default="items")
    parser.add_argument("--no-monitor", action="store_false", dest="monitor")
    result = benchmark(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                name: result[name]
                for name in (
                    "median_seconds",
                    "predicted_seconds",
                    "units_per_second",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
