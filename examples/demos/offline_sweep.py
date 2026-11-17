"""Compare hypothetical memory and transfer rates without a running service."""

import argparse

from shadowspill import PlanningInputs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("measurements", help="exported preparation measurements.zip")
parser.add_argument("--output", default="runs/demos/sweep.html")
options = parser.parse_args()

inputs = PlanningInputs.load(options.measurements)
sweep = inputs.sweep(device_gib=[12, 16, 24], transfer_scale=[0.5, 1, 2])
sweep.write_report(options.output)
print(f"Open {options.output} alongside its report directory.")
