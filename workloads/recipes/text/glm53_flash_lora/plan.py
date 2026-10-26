"""Run the shared text recipe plan phase."""

import sys

from workloads.recipes.text.phases.plan import run

if __name__ == "__main__":
    run(sys.argv[1])
