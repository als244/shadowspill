"""Run the shared text recipe profile phase."""

import sys

from workloads.recipes.text.phases.profile import run

if __name__ == "__main__":
    run(sys.argv[1])
