"""Module entry point for the shared build worker."""

import sys

from workloads.recipes.training.build import run

if __name__ == "__main__":
    run(sys.argv[1])
