"""CPU Program workers search orderings and jointly admit train/evaluation."""

import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from shadowspill.training.preparation.phases import settings

from .candidates import build_candidates, failures


def _candidate(request):
    """Load, plan and serialize one Program's ordering cases in a CPU worker."""
    from shadowspill import (
        CommunicationInfo,
        MemoryBudgets,
        PoolMemoryProfile,
        ProfiledProgram,
        plan_program,
    )
    from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
    from shadowspill.planning.shared import shared_memory
    from shadowspill.step import StepDataOrdering

    from .search import search_options

    root, options, candidate = request
    started = time.monotonic()
    path = candidate["path"]
    inputs = dict(
        memory_budgets=MemoryBudgets(
            int(options["execution_gib"] * (1 << 30)),
            int(options["spill_gib"] * (1 << 30)),
        ),
        search_options=search_options(options),
        pool_memory=PoolMemoryProfile.load(options["pool_memory"])
        if options.get("pool_memory")
        else None,
    )
    at = root / "profiled" / path
    evaluation = None
    rows = []
    try:
        if options.get("eval_every", 2):
            evaluation = plan_program(
                ProfiledProgram.load(at / "evaluation.ssp"),
                communication_info=CommunicationInfo.load(
                    root / "communication" / path / "evaluation.ssp"
                ),
                **inputs,
            )
    except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
        rows.append({**candidate, "evaluation_error": str(error)})
    else:
        profiled = ProfiledProgram.load(at / "training.ssp")
        communication = CommunicationInfo.load(
            root / "communication" / path / "training.ssp"
        )
        count = candidate["num_examples"]
        saved_evaluation = None
        for depth in (d for d in range(1, count + 1) if count % d == 0):
            ordering = StepDataOrdering(depth=depth, breadth=count // depth)
            row = {**candidate, "ordering": ordering.label}
            print(f"planning {row}", flush=True)
            try:
                training = plan_program(
                    profiled,
                    ordering=ordering,
                    communication_info=communication,
                    **inputs,
                )
                if evaluation is not None:
                    shared_memory(training, {"evaluation": evaluation})
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                row["error"] = str(error)
            else:
                row["duration_seconds"] = training.summary["duration_seconds"]
                destination = root / "plans/candidates" / path / ordering.label
                training.save(destination / "training.ssp")
                if evaluation is not None:
                    if saved_evaluation is None:
                        saved_evaluation = destination / "evaluation.ssp"
                        evaluation.save(saved_evaluation)
                    else:
                        shutil.copyfile(
                            saved_evaluation, destination / "evaluation.ssp"
                        )
            rows.append(row)
    return rows, dict(
        path=path,
        pid=os.getpid(),
        seconds=time.monotonic() - started,
        cases=len(rows),
        admitted=sum("duration_seconds" in row for row in rows),
    )


def _search(root, options, candidates):
    rejected = failures(root)
    rows = {
        index: [{**candidate, "failure": rejected[candidate["path"]]}]
        for index, candidate in enumerate(candidates)
        if candidate["path"] in rejected
    }
    pending = [
        (index, (root, options, candidate))
        for index, candidate in enumerate(candidates)
        if index not in rows
    ]
    workers = min(options.get("program_workers", 8), len(pending))
    work = []
    resources = dict(
        program_workers=workers,
        search_workers_per_program=options.get("search_workers", 8),
    )
    print(f"planning CPU concurrency: {resources}", flush=True)
    if workers <= 1:
        for index, request in pending:
            rows[index], observation = _candidate(request)
            work.append(observation)
    else:
        # Paths and small configs cross process boundaries; large Plans stay in
        # their worker and are saved directly to disjoint Program directories.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(_candidate, request): index for index, request in pending
            }
            try:
                for future in as_completed(futures):
                    rows[futures[future]], observation = future.result()
                    work.append(observation)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    # Preserve candidate/tie order independently of worker completion order.
    return (
        [row for index in range(len(candidates)) for row in rows[index]],
        {**resources, "programs": sorted(work, key=lambda row: row["path"])},
    )


def run(directory):
    from shadowspill import Plan, write_report
    from shadowspill.errors import PlanInfeasibleError

    root, options = settings(directory)
    started = time.monotonic()
    # A failed replan must not leave a runnable selection from the old budget.
    for name in ("selected.json", "search.json", "planning-work.json"):
        (root / name).unlink(missing_ok=True)
    for name in ("plans", "reports"):
        if (root / name).exists():
            shutil.rmtree(root / name)
    rows, work = _search(root, options, build_candidates(options))
    (root / "search.json").write_text(json.dumps(rows, indent=2) + "\n")
    admitted = [row for row in rows if "duration_seconds" in row]
    if not admitted:
        work["seconds"] = time.monotonic() - started
        (root / "planning-work.json").write_text(json.dumps(work, indent=2) + "\n")
        raise PlanInfeasibleError(
            "no training/evaluation pair fits; see search.json",
            kind="recipe_candidates",
        )
    choice = min(admitted, key=lambda row: row["duration_seconds"])
    chosen = root / "plans/candidates" / choice["path"] / choice["ordering"]
    for name in ("training", "evaluation"):
        source = chosen / f"{name}.ssp"
        if not source.exists():
            continue
        # The worker already serialized this exact winner. Copy its archive
        # instead of converting/compressing the complete Plan a second time.
        shutil.copyfile(source, root / f"plans/{name}.ssp")
        write_report(
            Plan.load(source), root / f"reports/{name}", plots=options["plots"]
        )
    shutil.copyfile(
        root / "programs" / choice["path"] / "metadata.json", root / "model.json"
    )
    work["seconds"] = time.monotonic() - started
    (root / "planning-work.json").write_text(json.dumps(work, indent=2) + "\n")
    (root / "selected.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"selected {choice}", flush=True)


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
