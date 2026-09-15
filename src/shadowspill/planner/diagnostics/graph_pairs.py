"""What each graph-pair selection cost, read off the program and the search.

A search settles one selection at a time -- one choice of graph-pair option
per task-alternative group -- and answers with the best plan across all of
them. Keeping only that answer hides what the choice cost, so a search's
diagnostics keep every resolved program it evaluated, and these records turn
them into one line each: whether the winner beat the alternatives by a hair or
by a factor, and whether the others were slower or simply would not fit.

Everything here is derivable without a second simulation. The costs are read
off the program rather than the simulator: a group's option names the tasks
it activates, and a task names its profile, so both the cheapest total and a
selection's total are sums over the same table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from shadowspill.ir import ShadowSpillProgram

from .json import _integer, _mapping, _number, _optional_number, _string

if TYPE_CHECKING:
    from shadowspill.planner.result import ProgramPlanResult


@dataclass(frozen=True, slots=True)
class AlternativeCosts:
    """Every task-alternative option's compute, and the floor beneath them.

    One table for the plan summary and the graph-pair outcomes to read, so
    the two agree by construction on what a selection costs and what the
    cheapest possible selection would.
    """

    #: Compute each option asks for, by ``(group_id, option_id)``.
    option_ns: Mapping[tuple[str, str], int]
    #: The cheapest option of each group.
    cheapest_ns: Mapping[str, int]
    #: Compute of the tasks no option activates, paid under every selection.
    fixed_ns: int

    @classmethod
    def from_program(cls, program: ShadowSpillProgram) -> AlternativeCosts:
        runtime_ns = {item.profile_id: item.runtime_ns for item in program.profiles}
        task_ns = {item.task_id: runtime_ns[item.profile_id] for item in program.tasks}
        variant_tasks: set[str] = set()
        option_ns: dict[tuple[str, str], int] = {}
        cheapest_ns: dict[str, int] = {}
        for group in program.task_alternative_groups:
            for option in group.options:
                variant_tasks.update(option.active_task_ids)
                option_ns[(group.group_id, option.option_id)] = sum(
                    task_ns[task_id] for task_id in option.active_task_ids
                )
            cheapest_ns[group.group_id] = min(
                option_ns[(group.group_id, option.option_id)]
                for option in group.options
            )
        fixed_ns = sum(
            task_ns[item.task_id]
            for item in program.tasks
            if item.task_id not in variant_tasks
        )
        return cls(MappingProxyType(option_ns), MappingProxyType(cheapest_ns), fixed_ns)

    @property
    def floor_ns(self) -> int:
        """Every group at its cheapest: the compute floor the program admits."""

        return self.fixed_ns + sum(self.cheapest_ns.values())

    def selected_ns(self, choices: Mapping[str, str]) -> int:
        """The compute one selection asks for, by the option chosen per group."""

        return self.fixed_ns + sum(
            self.option_ns[(group_id, option_id)]
            for group_id, option_id in choices.items()
        )

    def recomputing(self, choices: Mapping[str, str]) -> int:
        """Groups whose chosen option costs strictly more than their cheapest.

        That is what counts as a recomputation selection, whatever the
        options are named.
        """

        return sum(
            1
            for group_id, option_id in choices.items()
            if self.option_ns[(group_id, option_id)] > self.cheapest_ns[group_id]
        )


@dataclass(frozen=True, slots=True)
class GraphPairOutcome:
    """The best plan the search found under one graph-pair selection.

    ``selected_compute_seconds`` and ``unconstrained_seconds`` come from the
    program's task profiles and this selection's own choices, so the split
    below needs only the makespan the search already recorded.
    """

    selection_id: str
    #: Groups this selection asked to recompute rather than save, out of the
    #: graph-pair groups the program has. This is the "level" the ladder
    #: of selections walks.
    recompute_groups: int
    group_count: int
    #: The makespan of this selection's best plan, or `None` when no policy
    #: produced a plan that fits.
    makespan_seconds: float | None
    #: Compute this selection asks for, and the cheapest any selection could
    #: ask for, both as the sum of the selected tasks' profiles.
    selected_compute_seconds: float
    unconstrained_seconds: float
    valid_candidate_count: int
    candidate_count: int
    #: What this selection's own best plan moves. Zero when it placed
    #: nothing, and on a plan read back from a store written before these
    #: were recorded.
    fetched_bytes: int
    evicted_bytes: int

    @property
    def recomputation_overhead_seconds(self) -> float:
        """Compute this selection spends above the cheapest possible."""

        return self.selected_compute_seconds - self.unconstrained_seconds

    @property
    def waiting_seconds(self) -> float | None:
        """Everything in the step that is not compute.

        Waiting between tasks and the terminal writeback together, because
        separating them needs the span of this selection's plan and the
        search records only its makespan.
        """

        if self.makespan_seconds is None:
            return None
        return self.makespan_seconds - self.selected_compute_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "selection_id": self.selection_id,
            "recompute_groups": self.recompute_groups,
            "group_count": self.group_count,
            "makespan_seconds": self.makespan_seconds,
            "selected_compute_seconds": self.selected_compute_seconds,
            "unconstrained_seconds": self.unconstrained_seconds,
            "valid_candidate_count": self.valid_candidate_count,
            "candidate_count": self.candidate_count,
            "fetched_bytes": self.fetched_bytes,
            "evicted_bytes": self.evicted_bytes,
        }

    @classmethod
    def from_dict(cls, value: object, path: str) -> GraphPairOutcome:
        """Read back what :meth:`as_dict` wrote."""

        record = _mapping(value, path)
        return cls(
            selection_id=_string(record["selection_id"], f"{path}.selection_id"),
            recompute_groups=_integer(
                record["recompute_groups"], f"{path}.recompute_groups"
            ),
            group_count=_integer(record["group_count"], f"{path}.group_count"),
            makespan_seconds=_optional_number(
                record.get("makespan_seconds"), f"{path}.makespan_seconds"
            ),
            selected_compute_seconds=_number(
                record["selected_compute_seconds"],
                f"{path}.selected_compute_seconds",
            ),
            unconstrained_seconds=_number(
                record["unconstrained_seconds"], f"{path}.unconstrained_seconds"
            ),
            valid_candidate_count=_integer(
                record.get("valid_candidate_count", 0),
                f"{path}.valid_candidate_count",
            ),
            candidate_count=_integer(
                record.get("candidate_count", 0), f"{path}.candidate_count"
            ),
            fetched_bytes=_integer(
                record.get("fetched_bytes", 0), f"{path}.fetched_bytes"
            ),
            evicted_bytes=_integer(
                record.get("evicted_bytes", 0), f"{path}.evicted_bytes"
            ),
        )


def graph_pair_outcomes(result: ProgramPlanResult) -> tuple[GraphPairOutcome, ...]:
    """One record per graph-pair selection the search evaluated, ordered by
    how many groups recompute."""

    costs = AlternativeCosts.from_program(result.program)
    outcomes = []
    for problem in result.diagnostics.resolved_programs:
        chosen = {item.group_id: item.option_id for item in problem.choices}
        statuses = [item.status for item in problem.candidate_evaluations]
        outcomes.append(
            GraphPairOutcome(
                selection_id=problem.selection_id,
                recompute_groups=costs.recomputing(chosen),
                group_count=len(chosen),
                makespan_seconds=(
                    None
                    if problem.selected_makespan_ns is None
                    else problem.selected_makespan_ns / 1e9
                ),
                selected_compute_seconds=costs.selected_ns(chosen) / 1e9,
                unconstrained_seconds=costs.floor_ns / 1e9,
                valid_candidate_count=sum(1 for item in statuses if item == "valid"),
                candidate_count=len(statuses),
                fetched_bytes=problem.fetched_bytes,
                evicted_bytes=problem.evicted_bytes,
            )
        )
    return tuple(sorted(outcomes, key=lambda item: item.recompute_groups))


__all__ = ["AlternativeCosts", "GraphPairOutcome", "graph_pair_outcomes"]
