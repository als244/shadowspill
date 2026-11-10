"""Module entry point for the shared plan worker."""

import sys

from workloads.recipes.training.plan import run

if __name__ == "__main__":
    run(sys.argv[1])
