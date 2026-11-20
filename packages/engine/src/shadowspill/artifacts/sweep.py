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
    """One candidate planned within one budget, in `SweepResult.cases`.

    Attributes
    ----------
    candidate : str
        The candidate's name in the ``programs`` given to `sweep_programs`.
    memory_budgets : MemoryBudgets
        The budgets it was planned within.
    plan : Plan or None
        The Plan; None when planning failed.
    failure : mapping
        Empty when ``plan`` is set. Otherwise the error's ``type``,
        ``message`` and ``diagnostics``, with its ``kind``, ``device_id``,
        ``boundary_task_id``, ``required_bytes`` and ``capacity_bytes`` when
        it has them.
    """

    candidate: str
    memory_budgets: MemoryBudgets
    plan: Plan | None = None
    failure: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure", freeze(self.failure))


@dataclass(frozen=True)
class SweepResult(Artifact):
    """Every candidate and budget `sweep_programs` planned, failures included.

    Attributes
    ----------
    cases : tuple of SweepCase
        One case per budget and candidate: budgets in the order given, and
        within each, candidates in the order given.
    """

    artifact_kind: ClassVar[str] = "sweep_result"
    cases: tuple[SweepCase, ...]

    @property
    def best_per_budget(self) -> Mapping[MemoryBudgets, Plan]:
        """The Plan with the shortest simulated time for each budget, read-only.

        A budget without a Plan is absent; of equally fast Plans, the first
        case's is kept.
        """
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
        """Return the cases as plain data.

        Returns
        -------
        dict
            ``cases``, each with its candidate, budgets, Plan in
            `Plan.to_dict` form or None, and failure.
        """
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
        """Rebuild a result from `to_dict` data and its binary payloads.

        Parameters
        ----------
        value : dict
            The result, in `to_dict` form.
        payloads : mapping of str to bytes, optional
            The Plans' binary payloads, by SHA-256 hex digest.

        Returns
        -------
        SweepResult
            The result.

        Raises
        ------
        ValueError
            If a payload is missing, or a Plan is invalid.
        """
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
