"""Module entry point for the shared execute worker."""

import sys

from workloads.recipes.training.execute import run

if __name__ == "__main__":
    run(sys.argv[1])
