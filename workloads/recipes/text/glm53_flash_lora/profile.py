"""Module entry point for the shared profile worker."""

import sys

from workloads.recipes.training.profile import run

if __name__ == "__main__":
    run(sys.argv[1])
