"""Opt-in export of verified distributed inputs for CPU-only replanning.

The exported programs already contain the distributed search's conservative
profiles, collective costs and allocation contracts. No tensor or callable is
saved, and the observer never changes a search answer.
"""

from __future__ import annotations

import functools
import hashlib
import json
import traceback
from pathlib import Path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    temporary.replace(path)


def record_requests(directory, problems, requests, *, incumbents, options):
    """Keep each exact machine question and deduplicate its profiled program."""
    directory = Path(directory)
    if requests is None:
        raise ValueError("distributed symmetry was not verified for this geometry")
    saved = {}
    equivalents = {}
    local_identities = {index: problem.digest for index, problem in enumerate(problems)}
    for (index, budget), (problem, transfer, shared, evidence) in requests.items():
        if shared is None:
            raise ValueError(
                "cannot replan an unverified distributed input as one rank"
            )
        # Shared search constructs a distinct immutable problem for each
        # budget even when its program and capacity contract are identical.
        # Compare complete values before serializing these large programs.
        key = id(problem)
        identity = saved.get(key)
        if identity is None:
            identity = next(
                (
                    digest
                    for prior, digest in equivalents.get(index, ())
                    if problem == prior
                ),
                None,
            )
            if identity is None:
                encoded = problem.to_json()
                identity = hashlib.sha256(encoded.encode()).hexdigest()
                equivalents.setdefault(index, []).append((problem, identity))
                destination = directory / "problems" / f"{identity}.json"
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".json.tmp")
                    temporary.write_text(encoded + "\n")
                    temporary.replace(destination)
            saved[key] = identity
        write(
            directory
            / "requests"
            / local_identities[index]
            / f"{budget[0]}-{budget[1]}.json",
            {
                "schema": "pressurefit_search_input/v1",
                "problem": f"problems/{identity}.json",
                "problem_digest": identity,
                "execution_budget_bytes": budget[0],
                "spill_budget_bytes": budget[1],
                "transfer_bandwidths": transfer.to_dict(),
                "incumbents": incumbents,
                "search_options": options,
                "distributed_evidence": evidence,
            },
        )


def install(directory):
    """Install on the reporting rank before quickstart search; return uninstall.

    Export failures are retained for the offline replay to reject. They must not
    change feasibility or discard the GPU benchmark that produced the profiles.
    """
    from shadowspill.pytorch.distributed import _sweep
    from shadowspill.pytorch.step_search.sweep import _Sweep

    directory = Path(directory)
    original_requests = _sweep._verified_requests
    original_build = _Sweep._record_build

    def observe(operation):
        try:
            operation()
        except Exception:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "export-errors.log").open("a") as stream:
                stream.write(traceback.format_exc() + "\n")
            print(f"Search input export failed: {directory}", flush=True)

    @functools.wraps(original_requests)
    def verified(planner, problems, budgets, *, incumbents):
        result = original_requests(planner, problems, budgets, incumbents=incumbents)
        observe(
            lambda: record_requests(
                directory,
                problems,
                result,
                incumbents=incumbents,
                options=None
                if planner.search_options is None
                else planner.search_options.to_dict(),
            )
        )
        return result

    @functools.wraps(original_build)
    def built(self, candidate, accumulation, ordering, step, build_seconds):
        result = original_build(
            self, candidate, accumulation, ordering, step, build_seconds
        )
        observe(
            lambda: write(
                directory / "geometries" / f"{step.digest}.json",
                {
                    "step_program_digest": step.digest,
                    "local_problem_digest": step.problem.digest,
                    "candidate": candidate,
                    "ordering": ordering.to_dict(),
                    "accumulation_count": accumulation,
                },
            )
        )
        return result

    _sweep._verified_requests = verified
    _Sweep._record_build = built

    def uninstall():
        _sweep._verified_requests = original_requests
        _Sweep._record_build = original_build

    return uninstall
