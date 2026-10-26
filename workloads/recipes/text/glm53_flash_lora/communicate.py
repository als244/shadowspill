"""Run the shared text recipe communicate phase."""

import sys

from workloads.recipes.text.phases.communicate import run

if __name__ == "__main__":
    run(sys.argv[1])
