"""CPU traversal and collective consistency shared by preparation and saved reuse."""

from collections.abc import Mapping
from math import isqrt

from shadowspill.artifacts.plan import Plan
from shadowspill.errors import PlanInfeasibleError
from shadowspill.planning.collectives import collective_sequence, first_difference
from shadowspill.resources.description import resource_groups
from shadowspill.step.ordering import StepDataOrdering
from shadowspill.training.preparation.capture.templates import TemplateCandidate
from shadowspill.training.preparation.planning.records import WorkerPlans


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
    for role in ("main", "optimizer"):
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
