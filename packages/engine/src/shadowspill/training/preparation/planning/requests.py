"""Portable operation/count requests for the existing CPU planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from shadowspill.step.ordering import StepDataOrdering
from shadowspill.training.preparation.capture.templates import (
    TemplateInventory,
    TemplateUse,
)
from shadowspill.training.preparation.planning.identity import (
    candidate_identity,
    plan_identity,
)
from shadowspill.training.preparation.planning.validation import _orderings
from shadowspill_client._records import integer


@dataclass(frozen=True)
class PlanningRequest:
    rank: int
    use: TemplateUse
    ordering: StepDataOrdering | None

    @property
    def id(self) -> str:
        return plan_identity(self.use, self.ordering)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "use": asdict(self.use),
            "ordering": None if self.ordering is None else self.ordering.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlanningRequest:
        if set(value) != {"rank", "use", "ordering"} or set(value["use"]) != {
            "template",
            "kind",
            "count",
        }:
            raise ValueError("invalid exported planning request")
        use = TemplateUse(**value["use"])
        if use.kind not in {"step", "forward", "forward_backward", "optim_step"}:
            raise ValueError("unsupported exported operation")
        integer(use.count, "request count", minimum=1)
        return cls(
            integer(value["rank"], "logical rank"),
            use,
            None
            if value["ordering"] is None
            else StepDataOrdering.from_dict(value["ordering"], "ordering"),
        )


def planning_requests(inventory: TemplateInventory) -> dict[str, Any]:
    """Describe alternatives without building models or selecting a winner."""
    requests: dict[str, Any] = {}
    candidates = {}
    descriptions = {}
    for rows in inventory.families.values():
        for candidate in rows:
            for ordering in _orderings(candidate):
                members = []
                for rank, worker in candidate.workers.items():
                    for use in (worker.main, worker.optimizer):
                        if use is None:
                            continue
                        request = PlanningRequest(
                            rank,
                            use,
                            ordering
                            if use.kind in ("step", "forward_backward")
                            else None,
                        )
                        if (
                            request.id in requests
                            and requests[request.id] != request.to_dict()
                        ):
                            raise ValueError("planning request identity collision")
                        requests[request.id] = request.to_dict()
                        members.append(request.id)
                identity = candidate_identity(candidate, ordering)
                candidates[identity] = members
                shape = candidate.layout.candidate
                setup = inventory.layouts.setup
                descriptions[identity] = {
                    "operation": candidate.layout.operation.to_dict(),
                    "batch": setup.config.batches[shape.batch].to_dict(),
                    "capacities": dict(shape.schedule.capacities(setup.context.schema)),
                    "build": setup.config.preparation.cases[shape.case].to_dict(),
                    "microbatch_count": shape.count,
                    "ordering": None if ordering is None else ordering.to_dict(),
                }
    return {
        "requests": requests,
        "candidates": candidates,
        "descriptions": descriptions,
    }
