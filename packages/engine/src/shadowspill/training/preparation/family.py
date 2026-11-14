"""Plan the requested count families through the existing CPU planner and admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isqrt
from types import MappingProxyType

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.pool_memory import PoolMemoryProfile
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication.info import CommunicationInfo
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planner.search import SearchOptions
from shadowspill.planning.collectives import collective_sequence, first_difference
from shadowspill.planning.plan import plan_program
from shadowspill.profiling.apply import apply_profiles
from shadowspill.resources.description import resource_groups
from shadowspill.step.ordering import StepDataOrdering
from shadowspill_client._records import integer

from .family_records import PlannedCandidate, PlanRejection, PreparedPlans, WorkerPlans
from .family_selection import select_family
from .templates import (
    TemplateCandidate,
    TemplateInventory,
    TemplateUse,
    instantiate_template,
    validate_template,
)


def _orderings(candidate: TemplateCandidate) -> tuple[StepDataOrdering | None, ...]:
    if candidate.layout.operation.operation == "forward":
        return (None,)
    counts = {worker.main.count for worker in candidate.workers.values()}
    if len(counts) != 1:
        raise ValueError("training traversal requires coordinated worker counts")
    count = next(iter(counts))
    divisors = {
        factor
        for d in range(1, isqrt(count) + 1)
        if count % d == 0
        for factor in (d, count // d)
    }
    return tuple(StepDataOrdering(depth, count // depth) for depth in sorted(divisors))


def _collectives(
    workers: Mapping[int, WorkerPlans], plans: Mapping[int, Mapping[str, Plan]]
) -> None:
    for role in ("main", "masked", "optimizer"):
        paths = {
            rank: plans[rank][name]
            for rank, worker in workers.items()
            if (name := getattr(worker, role)) is not None
        }
        if not paths:
            continue
        for accumulated in (False, True):
            selected = {
                rank: path._accumulation_plan if accumulated else path
                for rank, path in paths.items()
            }
            if all(p is None for p in selected.values()):
                continue
            if any(p is None for p in selected.values()):
                raise ValueError("workers disagree on gradient accumulation paths")
            actual = {rank: path for rank, path in selected.items() if path is not None}
            difference = first_difference(
                role, [collective_sequence(actual[r]) for r in sorted(actual)]
            )
            if difference is not None:
                raise PlanInfeasibleError(difference, kind="prepared_collective_order")
            # The existing issue-order helper cannot see calls inside provider
            # kernels. Matching finalized task sequences within each provider
            # group is a conservative sufficient check; differing opaque graphs
            # require a more explicit provider protocol, never a guessed match.
            groups = {
                group
                for p in actual.values()
                for group in resource_groups(p.program.resources)
            }
            for group in groups:
                sequences = []
                for rank in group:
                    plan = actual[rank]
                    definitions = {
                        d.profile_key: d.compatibility_digest
                        for d in plan.program.task_definitions
                    }
                    # Costing may append workspace suffixes to profile IDs.
                    # Compare the original finalized form, not a cost-only ID.
                    originals = {
                        task.task_id: task.profile_id for task in plan.program.tasks
                    }
                    sequences.append(
                        (
                            {n: r.to_dict() for n, r in plan.program.resources.items()},
                            tuple(
                                definitions[originals[t.task_id]]
                                for t in plan._costed_program.selected_tasks(
                                    plan.selections
                                )
                            ),
                        )
                    )
                if any(value != sequences[0] for value in sequences[1:]):
                    raise ValueError(
                        "opaque provider collective graphs differ "
                        "across cooperating workers"
                    )


def plan_family(
    inventory: TemplateInventory,
    profiled: Mapping[str, ProfiledProgram],
    communication: Mapping[str, CommunicationInfo],
    *,
    pool_memory: Mapping[int, PoolMemoryProfile] | None = None,
    max_plan_attempts: int = 16384,
    max_family_attempts: int = 4096,
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
    for key, template in inventory.templates.items():
        source = profiled[key]
        validate_template(template, source.program)
        if (
            source.metadata.get("preparation_template") != key
            or plain(source.metadata.get("profiling_scenario"))
            != template.signature()["scenario"]
            or plain(source.metadata.get("profiling_options"))
            != setup.config.preparation.profiling.to_dict()
        ):
            raise ValueError(
                "saved profiling policy/scenario differs from the prepared template"
            )
        if source.estimated_memory:
            raise ValueError(
                "prepared operation family requires measured memory evidence"
            )
    candidates = [row for rows in inventory.families.values() for row in rows]
    attempts = sum(
        len(_orderings(c))
        * sum(
            1 + (w.masked is not None) + (w.optimizer is not None)
            for w in c.workers.values()
        )
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
            return found
        source = profiled[use.template]
        program = instantiate_template(template, use, source.program)
        try:
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
        name = digest(
            {
                "template": use.template,
                "kind": use.kind,
                "count": use.count,
                "ordering": key[1],
            }
        )
        plans[rank][name] = result
        cached[key] = name
        return name

    admitted, rejected = [], []
    for candidate in candidates:
        shape = candidate.layout.candidate
        operation = candidate.layout.operation
        for ordering in _orderings(candidate):
            label = None if ordering is None else ordering.label
            identity = digest(
                {
                    "operation": operation.to_dict(),
                    "batch": shape.batch,
                    "case": shape.case,
                    "count": shape.count,
                    "ordering": label,
                }
            )
            try:
                workers = {
                    rank: WorkerPlans(
                        materialize(worker.main, rank, ordering),
                        None
                        if worker.masked is None
                        else materialize(worker.masked, rank, None),
                        None
                        if worker.optimizer is None
                        else materialize(worker.optimizer, rank, None),
                    )
                    for rank, worker in sorted(candidate.workers.items())
                }
                _collectives(workers, plans)
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
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
            admitted.append(
                PlannedCandidate(
                    identity, candidate, ordering, MappingProxyType(workers), duration
                )
            )
    return select_family(
        inventory, admitted, plans, rejected, max_family_attempts=max_family_attempts
    )
