"""Prepare saved phases, then launch fresh baseline measurement workers."""

import argparse
import os
import shutil
import subprocess
import traceback
from pathlib import Path

from benchmarking.processes import worker_command
from benchmarking.quickstart.phases.common import read, write
from shadowspill.errors import PlanInfeasibleError

from .preparation import prepare
from .settings import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args(argv)
    output, case_file = args.output.resolve(), args.case.resolve()
    case = read(case_file)
    settings = Settings.from_dict(case["settings"])
    os.environ["OMP_NUM_THREADS"] = str(settings.cpu_threads)
    try:
        root, preparation = prepare(case, settings, case_file, output)
        if "preparation" in case:
            write(
                output / "preparation-result.json",
                {"status": "completed", **preparation},
            )
            return 0
        budget = int(settings.execution_gib * (1 << 30))
        selected = next(
            (
                entry
                for entry in read(root / "plans/index.json")
                if entry["budget"] == budget
            ),
            None,
        )
        if selected is None:
            raise PlanInfeasibleError(
                f"no candidate fits {settings.execution_gib:g} GiB; "
                f"see {root / 'report'}",
                kind="baseline_budget",
            )
        # Each result keeps its executable even if the shared folder is replanned.
        saved_plan = output / "plan.ssp"
        shutil.copyfile(root / "plans" / selected["file"], saved_plan)
        case["phase_execution"] = {
            **preparation,
            "selected": selected,
            "plan": str(saved_plan),
        }
        write(case_file, case)
    except Exception as error:
        write(
            output / "rank-00000/failure.json",
            {
                "rank": 0,
                "stage": "preparation",
                "status": "cuda_out_of_memory"
                if isinstance(error, PlanInfeasibleError)
                else "failed",
                "boundary": isinstance(error, PlanInfeasibleError),
                "exception_type": type(error).__name__,
                "error": traceback.format_exc(),
            },
        )
        raise
    world = case["description"]["geometry"]["world_size"]
    return subprocess.call(
        worker_command(
            "baselines.engines.shadowspill.execute",
            ["--output", output, "--case", case_file],
            ranks=world,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
