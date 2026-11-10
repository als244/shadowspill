"""Module entry point for the shared communicate worker."""

import sys

from workloads.recipes.training.communicate import run

if __name__ == "__main__":
    run(sys.argv[1])
