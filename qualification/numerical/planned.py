"""One planned case end to end: run it, compare it, record it, judge it."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from qualification.pipeline.process import profile_and_plan, run_phase
from qualification.pipeline.retention import remove_execution_files

from .compare import compare_planned_run
from .evidence import qualification_artifact
from .execution import run_planned_case
from .prepare import load_request, save_request
from .request import PlannedRequest
from .verdict import qualification_failed, record_verdict


def planned_worker(request: PlannedRequest) -> None:
    """Orchestrate independent preparation and execution processes."""
    directory = request.result_path.with_suffix(".phases").resolve()
    save_request(request, directory)
    run_phase(directory, "build", "qualification.numerical.prepare", directory)
    profile_and_plan(directory)
    run_phase(directory, "execute", "qualification.numerical.planned", directory)
    if not request.retain_execution_files:
        remove_execution_files(directory)


def execute(directory: Path) -> None:
    request = load_request(directory)
    run = run_planned_case(request, directory)
    comparison = compare_planned_run(request, run)
    started = time.perf_counter()
    result = qualification_artifact(request, run, comparison)
    failures = record_verdict(result, request, run, comparison)
    request.result_path.parent.mkdir(parents=True, exist_ok=True)
    request.result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"shadowspill {request.case.model_implementation}/{request.case.family} "
        f"wrote {request.result_path.name}, {len(failures)} failure(s): "
        f"{time.perf_counter() - started:.3f}s",
        flush=True,
    )
    if failures:
        raise qualification_failed(request, failures)


if __name__ == "__main__":
    execute(Path(sys.argv[1]))
