"""One planned case end to end: run it, compare it, record it, judge it."""

from __future__ import annotations

import json
import time

from .compare import compare_planned_run
from .evidence import qualification_artifact
from .request import PlannedRequest
from .run import run_planned_case
from .verdict import qualification_failed, record_verdict


def planned_worker(request: PlannedRequest) -> None:
    """Qualify one case, write its artifact, and fail if any check did."""

    run = run_planned_case(request)
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
