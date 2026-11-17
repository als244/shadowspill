"""Concurrent variant searches with shared homogeneous-worker results."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication.info import CommunicationInfo
from shadowspill.diagnostics.progress import Progress, event, phase
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.memory_search.algorithm import SearchOptions
from shadowspill.planning.plan import plan_program
from shadowspill.profiling.apply import apply_profiles
from shadowspill_client._records import plain
from shadowspill_client.protocol.training import ResolvedTraining

from .planning_requests import PlanningRequest, planning_requests
from .shared_planning import representative_sources, representative_use
from .templates import TemplateInventory, instantiate_program

SearchResult = Plan | PlanInfeasibleError | PlanSearchExhaustedError


@lru_cache(maxsize=1)
def _measurements(root: str, key: str) -> tuple[ProfiledProgram, CommunicationInfo]:
    # A process retains at most one immutable capture. Variants of that capture
    # reuse it; large Program trees never cross the multiprocessing pipe.
    directory = Path(root)
    return (
        ProfiledProgram.load(directory / "calibrated" / f"{key}.ssp"),
        CommunicationInfo.load(directory / "communication" / f"{key}.ssp"),
    )


def _search(root: str, settings: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    request = PlanningRequest.from_dict(row)
    source, communication = _measurements(root, request.use.template)
    search = replace(
        SearchOptions.from_dict(settings["search"], "search"),
        workers=settings["threads"],
    )
    ordering = request.ordering
    title = (
        f"Worker {request.rank} · {request.use.kind} · "
        f"{request.use.count} microbatches · "
        f"{ordering.label if ordering is not None else 'Default ordering'}"
    )
    with (
        Progress(lambda line: print(line, flush=True)).activate(),
        phase(title) as outcome,
    ):
        try:
            program = instantiate_program(request.use, source.program)
            plan = plan_program(
                apply_profiles(program, source.profiles),
                memory_budgets=MemoryBudgets.from_value(settings["budgets"]),
                communication_info=communication,
                search_options=search,
                ordering=ordering,
            )
        except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
            outcome.rejected = True
            event(str(error))
            return {
                "error": str(error),
                "exhausted": isinstance(error, PlanSearchExhaustedError),
                "kind": getattr(error, "kind", None),
            }
        if plan.simulation_only_reasons:
            raise ValueError("prepared Plan lacks physical admission")
        path = Path(root) / "searched" / request.use.template / f"{request.id}.ssp"
        plan.save(path)
        return {"path": str(path)}


def parallel_search(
    root: Path,
    inventory: TemplateInventory,
    *,
    sources: Mapping[str, ProfiledProgram] | None = None,
    communication: Mapping[str, CommunicationInfo] | None = None,
    max_processes: int = 8,
    max_plan_attempts: int = 16384,
) -> dict[str, SearchResult]:
    """Search unique variants across all distinct ranks under one CPU budget.

    Equivalent ranks use the same complete measured implementation and Plan.
    Every count/ordering is independently schedulable, including on a single
    GPU. Caller performs source checks and joint family admission. Workers use
    CPU APIs only; archives bound IPC size and preserve results after failures.
    """
    requests = {
        key: PlanningRequest.from_dict(row)
        for key, row in planning_requests(inventory)["requests"].items()
    }
    if sources is None:
        sources = {
            key: ProfiledProgram.load(root / "calibrated" / f"{key}.ssp")
            for key in inventory.templates
        }
    if communication is None:
        communication = {
            key: CommunicationInfo.load(root / "communication" / f"{key}.ssp")
            for key in inventory.templates
        }
    return search_requests(
        root,
        inventory.layouts.setup.config,
        requests,
        sources,
        communication,
        max_processes=max_processes,
        max_plan_attempts=max_plan_attempts,
    )


def search_requests(
    root: Path,
    config: ResolvedTraining,
    requests: Mapping[str, PlanningRequest],
    sources: Mapping[str, ProfiledProgram],
    communication: Mapping[str, CommunicationInfo],
    *,
    max_processes: int = 8,
    max_plan_attempts: int = 16384,
) -> dict[str, SearchResult]:
    """Run already described requests; no installed workload imports are needed."""
    representatives = representative_sources(sources, communication=communication)
    aliases, unique = {}, {}
    for key, request in requests.items():
        use = representative_use(request.use, representatives)
        canonical = replace(
            request,
            use=use,
            rank=sources[use.template].program.bindings["preparation_template"][
                "binding"
            ]["logical_rank"],
        )
        aliases[key] = canonical.id
        unique[canonical.id] = canonical
    if len(unique) > max_plan_attempts:
        raise ValueError(
            f"planning expansion exceeds configured limit {max_plan_attempts}"
        )
    if not unique:
        return {}
    if type(max_processes) is not int or max_processes < 1:
        raise ValueError("max_processes must be positive")
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    planning = config.preparation.planning
    # Explicit workers is a total budget, not a multiplier per logical rank.
    budget = min(available, planning.workers or available)
    processes = min(max_processes, len(unique), budget)
    threads = max(1, budget // processes)
    settings = plain(
        {
            "search": planning.search,
            "threads": threads,
            "budgets": MemoryBudgets(
                int(config.memory.device_gib * (1 << 30)),
                int(config.memory.spill_gib * (1 << 30)),
            ).to_dict(),
        }
    )
    event(
        f"CPU planning: {processes} processes x {threads} search threads; "
        f"{len(unique)} unique variants for {len(requests)} worker requests "
        f"({len(requests) - len(unique)} shared)"
    )
    results: dict[str, SearchResult] = {}
    with ProcessPoolExecutor(
        max_workers=processes, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(_search, str(root), settings, row.to_dict()): key
            for key, row in unique.items()
        }
        for completed, future in enumerate(as_completed(futures), 1):
            key, result = futures[future], future.result()
            if "path" in result:
                results[key] = Plan.load(result["path"])
            elif result["exhausted"]:
                results[key] = PlanSearchExhaustedError(result["error"])
            else:
                results[key] = PlanInfeasibleError(result["error"], kind=result["kind"])
            event(f"Completed CPU variant searches: {completed}/{len(unique)}")
    return {key: results[canonical] for key, canonical in aliases.items()}
