"""Plans and diagnostic failures retained for every explicitly supplied case."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from ._values import freeze, plain
from .archive import Artifact
from .plan import Plan

if TYPE_CHECKING:
    from shadowspill.artifacts.budgets import MemoryBudgets


@dataclass(frozen=True)
class SweepCase:
    candidate: str
    memory_budgets: MemoryBudgets
    plan: Plan | None = None
    failure: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure", freeze(self.failure))


@dataclass(frozen=True)
class SweepResult(Artifact):
    artifact_kind: ClassVar[str] = "sweep_result"
    cases: tuple[SweepCase, ...]

    @property
    def best_per_budget(self) -> Mapping[MemoryBudgets, Plan]:
        best: dict[MemoryBudgets, Plan] = {}
        for case in self.cases:
            previous = best.get(case.memory_budgets)
            if case.plan is not None and (
                previous is None
                or case.plan.timeline.makespan_ns < previous.timeline.makespan_ns
            ):
                best[case.memory_budgets] = case.plan
        return MappingProxyType(best)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [
                {
                    "candidate": item.candidate,
                    "memory_budgets": item.memory_budgets.to_dict(),
                    "plan": None if item.plan is None else item.plan.to_dict(),
                    "failure": plain(item.failure),
                }
                for item in self.cases
            ]
        }

    def _archive_payloads(self) -> Mapping[str, bytes]:
        return {
            key: value
            for case in self.cases
            if case.plan is not None
            for key, value in case.plan._archive_payloads().items()
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> SweepResult:
        from shadowspill.artifacts.budgets import MemoryBudgets

        return cls(
            tuple(
                SweepCase(
                    item["candidate"],
                    MemoryBudgets(**item["memory_budgets"]),
                    None
                    if item["plan"] is None
                    else Plan.from_dict(item["plan"], payloads=payloads),
                    item["failure"],
                )
                for item in value["cases"]
            )
        )
