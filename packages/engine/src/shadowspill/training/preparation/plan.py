"""Run the plan phase of a resolved public training preparation job."""

from .job_phases import plan as run

if __name__ == "__main__":
    import sys

    run(sys.argv[1])
