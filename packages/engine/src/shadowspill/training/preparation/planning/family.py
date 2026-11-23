"""Plan the requested count families through the existing CPU planner and admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.pool_memory import PoolMemoryProfile
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication.info import CommunicationInfo
from shadowspill.diagnostics.progress import event, phase
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.memory_search.algorithm import SearchOptions
from shadowspill.planning.plan import plan_program
from shadowspill.profiling.apply import apply_profiles
from shadowspill.step.ordering import StepDataOrdering
from shadowspill.training.preparation.capture.templates import (
    TemplateInventory,
    TemplateUse,
    instantiate_template,
)
from shadowspill.training.preparation.planning.coordination import coordinate_workers
from shadowspill.training.preparation.planning.identity import (
    candidate_identity,
    execution_names,
    plan_identity,
    validate_profile,
)
from shadowspill.training.preparation.planning.records import (
    PlannedCandidate,
    PlanRejection,
    PreparedPlans,
    WorkerPlans,
)
from shadowspill.training.preparation.planning.selection import select_family
from shadowspill.training.preparation.planning.sharing import (
    representative_sources,
    representative_use,
)
from shadowspill.training.preparation.planning.validation import (
    _collectives,
    _orderings,
)
from shadowspill_client._records import integer


def plan_family(
    inventory: TemplateInventory,
    profiled: Mapping[str, ProfiledProgram],
    communication: Mapping[str, CommunicationInfo],
    *,
    pool_memory: Mapping[int, PoolMemoryProfile] | None = None,
    max_plan_attempts: int = 16384,
    max_family_attempts: int = 4096,
    prior_rejections: Sequence[PlanRejection] = (),
    search_results: Mapping[str, Plan | PlanInfeasibleError | PlanSearchExhaustedError]
    | None = None,
) -> PreparedPlans:
    """Admit all requested capabilities, preserving coverage after known failures.

    This never builds or profiles missing forms, lowers capacity, substitutes a
    simulation-only Plan, or constructs an execution worker. Saved profiles and
    communication evidence are required for the exact capture inventory.
    """
    integer(max_plan_attempts, "max_plan_attempts", minimum=1)
    integer(max_family_attempts, "max_family_attempts", minimum=1)
    if (
        profiled.keys() != inventory.templates.keys()
        or communication.keys() != inventory.templates.keys()
    ):
        raise ValueError(
            "provide profiles and communication for exactly the capture inventory"
        )
    setup = inventory.layouts.setup
    ranks = set(range(setup.context.parallelism.world_size))
    if pool_memory is not None and pool_memory.keys() != ranks:
        raise ValueError("pool memory evidence must name every logical worker")
    for key in inventory.templates:
        validate_profile(inventory, key, profiled[key])
    representatives = representative_sources(profiled, pool_memory, communication)
    candidates = [row for rows in inventory.families.values() for row in rows]
    attempts = len(
        {
            plan_identity(representative_use(use, representatives), ordering)
            for candidate in candidates
            for ordering in _orderings(candidate)
            for worker in candidate.workers.values()
            for use in (worker.main, worker.optimizer)
            if use is not None
        }
    )
    if attempts > max_plan_attempts:
        raise ValueError(
            f"planning expansion exceeds configured limit {max_plan_attempts}"
        )
    if search_results is not None:
        from shadowspill.training.preparation.planning.requests import planning_requests

        if search_results.keys() != planning_requests(inventory)["requests"].keys():
            raise ValueError("CPU search results must cover exactly the Plan requests")
    configuration = setup.config.preparation.planning
    assert configuration.search is not None and configuration.workers is not None
    search = replace(
        SearchOptions.from_dict(configuration.search, "planning.search"),
        workers=configuration.workers,
    )
    budgets = MemoryBudgets(
        int(setup.config.memory.device_gib * (1 << 30)),
        int(setup.config.memory.spill_gib * (1 << 30)),
    )
    plans: dict[int, dict[str, Plan]] = {rank: {} for rank in ranks}
    event(
        f"Search workers: {search.workers or 'Automatic'}; "
        f"logical workers: {len(ranks)}; "
        f"budgets: GPU {setup.config.memory.device_gib:g} GiB, "
        f"spill {setup.config.memory.spill_gib:g} GiB"
    )
    cached: dict[
        tuple[TemplateUse, str | None, int],
        Plan | PlanInfeasibleError | PlanSearchExhaustedError,
    ] = {}
    reservations: dict[str, int] = {}
    if search_results is None:
        from shadowspill.training.preparation.planning.requests import (
            PlanningRequest,
            planning_requests,
        )
        from shadowspill.training.preparation.planning.reservations import (
            spill_reservations,
        )

        planning = planning_requests(inventory)
        requests = {
            key: PlanningRequest.from_dict(row)
            for key, row in planning["requests"].items()
        }
        reservations = spill_reservations(
            requests,
            planning["candidates"],
            planning["descriptions"],
            {
                key: instantiate_template(
                    inventory.templates[request.use.template],
                    request.use,
                    profiled[request.use.template].program,
                )
                for key, request in requests.items()
            },
        )

    def materialize(
        use: TemplateUse, rank: int, ordering: StepDataOrdering | None
    ) -> str:
        order = ordering if use.kind in ("step", "forward_backward") else None
        template = inventory.templates[use.template]
        if rank != template.binding.logical_rank:
            raise ValueError("template use belongs to another logical worker")
        name = plan_identity(use, order)
        reserved = reservations.get(name, 0)
        use = representative_use(use, representatives)
        template = inventory.templates[use.template]
        key = (use, None if order is None else order.label, reserved)
        if key in cached:
            found = cached[key]
            if isinstance(found, Exception):
                raise found
            plans[rank][name] = found
            event(
                f"Worker {rank} · Share {use.kind} Plan "
                f"from worker {template.binding.logical_rank}"
            )
            return name
        try:
            if search_results is not None:
                searched = search_results[name]
                if isinstance(searched, Exception):
                    raise searched
                result = searched
            else:
                if reserved > budgets.spill_bytes:
                    raise PlanInfeasibleError(
                        f"the run's other operations keep {reserved} bytes of "
                        f"persistent spill; the budget is {budgets.spill_bytes}",
                        kind="spill_reservation",
                        required_bytes=reserved,
                        capacity_bytes=budgets.spill_bytes,
                    )
                source = profiled[use.template]
                program = instantiate_template(template, use, source.program)
                with phase(f"Worker {rank} · {use.kind} · {use.count} microbatches"):
                    result = plan_program(
                        apply_profiles(program, source.profiles),
                        memory_budgets=budgets,
                        communication_info=communication[use.template],
                        pool_memory=None
                        if pool_memory is None
                        else pool_memory[template.binding.logical_rank],
                        search_options=search,
                        ordering=order,
                        reserved_spill_bytes=reserved,
                    )
        except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
            cached[key] = error
            raise
        if result.simulation_only_reasons:
            raise ValueError(
                "prepared Plan lacks physical admission: "
                + "; ".join(result.simulation_only_reasons)
            )
        plans[rank][name] = result
        cached[key] = result
        return name

    admitted, rejected = [], list(prior_rejections)
    total = sum(len(_orderings(candidate)) for candidate in candidates)
    index = 0
    for candidate in candidates:
        shape = candidate.layout.candidate
        operation = candidate.layout.operation
        for ordering in _orderings(candidate):
            index += 1
            label = None if ordering is None else ordering.label
            identity = candidate_identity(candidate, ordering)
            size = setup.config.preparation.cases[shape.case].microbatch_size
            operation_title = operation.operation.replace("_", " ").capitalize()
            title = (
                f"Layout {index}/{total} · {operation_title} · "
                f"Batch {shape.batch + 1}, case {shape.case + 1} · "
                f"Microbatch {size:,} x {shape.count}"
            )
            with phase(title) as outcome:
                if ordering is not None:
                    event(
                        f"Ordering: {ordering.depth} passes x "
                        f"{ordering.breadth} microbatches; "
                        f"paired terminal: {ordering.pair_loss}; "
                        f"reverse backward: {ordering.reverse_breadth}"
                    )
                try:
                    workers = {
                        rank: WorkerPlans(
                            materialize(worker.main, rank, ordering),
                            None
                            if worker.optimizer is None
                            else materialize(worker.optimizer, rank, None),
                        )
                        for rank, worker in sorted(candidate.workers.items())
                    }
                    workers = coordinate_workers(
                        workers, plans, execution_names(candidate, ordering)
                    )
                    _collectives(workers, plans)
                except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                    outcome.rejected = True
                    event(f"Reason: {error}")
                    rejected.append(
                        PlanRejection(
                            identity,
                            operation,
                            shape.batch,
                            shape.case,
                            shape.count,
                            label,
                            "planning",
                            str(error),
                            isinstance(error, PlanSearchExhaustedError),
                        )
                    )
                    continue
                duration = max(
                    plans[rank][worker.main].timeline.makespan_ns / 1e9
                    for rank, worker in workers.items()
                )
                event(f"Estimated {operation.operation} time: {duration:.3f}s")
                admitted.append(
                    PlannedCandidate(
                        identity,
                        candidate,
                        ordering,
                        MappingProxyType(workers),
                        duration,
                    )
                )
    with phase("Select jointly compatible Plans"):
        prepared = select_family(
            inventory,
            admitted,
            plans,
            rejected,
            max_family_attempts=max_family_attempts,
        )
        event(
            f"Retained {sum(map(len, prepared.families.values()))} layout/orderings, "
            f"{sum(map(len, prepared.plans.values()))} worker Plans; "
            f"rejected {len(prepared.rejected)} candidates"
        )
        for rank, memory in sorted(prepared.memory.items()):
            event(
                f"Worker {rank} shared layout: "
                f"GPU {memory.execution_bytes / (1 << 30):.2f} GiB; "
                f"spill {memory.spill_bytes / (1 << 30):.2f} GiB"
            )
        return prepared
