"""Coordinate each invocation without changing reusable rank-local searches."""

from collections.abc import Mapping

from shadowspill.artifacts.plan import Plan
from shadowspill.diagnostics.progress import event, phase
from shadowspill.planning.coordination import coordinate_plans
from shadowspill.training.preparation.planning.records import WorkerPlans


def coordinate_workers(
    workers: Mapping[int, WorkerPlans],
    plans: dict[int, dict[str, Plan]],
    names: WorkerPlans,
) -> dict[int, WorkerPlans]:
    for role in ("main", "optimizer"):
        name = getattr(names, role)
        if name is None:
            continue
        if all(name in plans[rank] for rank in workers):
            continue
        sources = {
            rank: plans[rank][getattr(worker, role)] for rank, worker in workers.items()
        }
        with phase(f"Coordinate {role} collective order"):
            coordinated = coordinate_plans(sources)
            changed = sum(
                plan is not sources[rank] for rank, plan in coordinated.items()
            )
            if changed:
                event(
                    f"Shared logical schedule; {changed} worker layouts "
                    "replayed and physically admitted"
                )
            for rank, plan in coordinated.items():
                plans[rank][name] = plan
    return dict.fromkeys(workers, names)
