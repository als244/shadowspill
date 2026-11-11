"""Small convenience entry point; the Python API accepts the same controls."""

import argparse
import json

from orchestration.nsys import NsysCapture

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
    parser.add_argument(
        "--no-final-trace", action="store_false", dest="trace_final_step"
    )
    parser.add_argument("--nsys-steps", type=int, default=0, help="0 disables Nsight")
    parser.add_argument("--nsys-start-step", type=int, default=2)
    parser.add_argument(
        "--nsys-output-dir",
        help="common capture folder (default: OUTPUT_DIR/profiling)",
    )
    options = parser.parse_args()
    if options.nsys_steps < 0:
        parser.error("--nsys-steps must be nonnegative")
    result = benchmark(
        options.run_dir,
        output_dir=options.output_dir,
        steps=options.steps,
        warmup_steps=options.warmup_steps,
        units_per_step=options.units_per_step,
        unit_label=options.unit_label,
        monitor=options.monitor,
        trace_final_step=options.trace_final_step,
        nsys=NsysCapture(
            start_step=options.nsys_start_step,
            steps=options.nsys_steps,
            output_dir=options.nsys_output_dir,
        )
        if options.nsys_steps
        else None,
    )
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
