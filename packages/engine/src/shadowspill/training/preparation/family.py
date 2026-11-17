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
from shadowspill.planner.search import SearchOptions
from shadowspill.planning.plan import plan_program
from shadowspill.profiling.apply import apply_profiles
from shadowspill.step.ordering import StepDataOrdering
from shadowspill_client._records import integer

from .family_checks import _collectives, _orderings
from .family_identity import candidate_identity, plan_identity, validate_profile
from .family_records import PlannedCandidate, PlanRejection, PreparedPlans, WorkerPlans
from .family_selection import select_family
from .templates import (
    TemplateInventory,
    TemplateUse,
    instantiate_template,
)


def plan_family(
    inventory: TemplateInventory,
    profiled: Mapping[str, ProfiledProgram],
    communication: Mapping[str, CommunicationInfo],
    *,
    pool_memory: Mapping[int, PoolMemoryProfile] | None = None,
    max_plan_attempts: int = 16384,
    max_family_attempts: int = 4096,
    prior_rejections: Sequence[PlanRejection] = (),
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
    candidates = [row for rows in inventory.families.values() for row in rows]
    attempts = sum(
        len(_orderings(c))
        * sum(1 + (w.optimizer is not None) for w in c.workers.values())
        for c in candidates
    )
    if attempts > max_plan_attempts:
        raise ValueError(
            f"planning expansion exceeds configured limit {max_plan_attempts}"
        )
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
        tuple[TemplateUse, str | None],
        str | PlanInfeasibleError | PlanSearchExhaustedError,
    ] = {}

    def materialize(
        use: TemplateUse, rank: int, ordering: StepDataOrdering | None
    ) -> str:
        order = ordering if use.kind in ("step", "forward_backward") else None
        template = inventory.templates[use.template]
        if rank != template.binding.logical_rank:
            raise ValueError("template use belongs to another logical worker")
        key = (use, None if order is None else order.label)
        if key in cached:
            found = cached[key]
            if isinstance(found, Exception):
                raise found
            event(f"Worker {rank} · Reuse {use.kind} Plan {found[:12]}")
            return found
        source = profiled[use.template]
        program = instantiate_template(template, use, source.program)
        try:
            with phase(f"Worker {rank} · {use.kind} · {use.count} microbatches"):
                result = plan_program(
                    apply_profiles(program, source.profiles),
                    memory_budgets=budgets,
                    communication_info=communication[use.template],
                    pool_memory=None if pool_memory is None else pool_memory[rank],
                    search_options=search,
                    ordering=order,
                )
        except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
            cached[key] = error
            raise
        if result.simulation_only_reasons:
            raise ValueError(
                "prepared Plan lacks physical admission: "
                + "; ".join(result.simulation_only_reasons)
            )
        name = plan_identity(use, order)
        plans[rank][name] = result
        cached[key] = name
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
