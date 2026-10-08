"""Share verified CPU search work across complete ordering/budget sequences.

All capture and profiling has finished before entering here. Search ownership,
budget incumbents, and local admission stay together in one geometry's state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial

from shadowspill.errors import (
    AdmissionError,
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.planner.diagnostics import graph_pair_outcomes
from shadowspill.planner.diagnostics.plan import summarize_selected_plan
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.search.planner import _Answer, _Carried, _Planner
from shadowspill.search.refusals import _EXHAUSTED, _INFEASIBLE
from shadowspill.store import ArtifactStore

from . import current
from ._control import Control
from ._plan_exchange import pack, received_plan, record_plan, shared_problem
from ._selection import record_decision


def _verified_requests(planner, problems, budgets, *, incumbents):
    """Verify the whole geometry before assigning any CPU work."""
    options = (
        None if planner.search_options is None else planner.search_options.to_dict()
    )
    requests = {
        (index, budget): shared_problem(
            problem,
            *budget,
            planner.transfer_bandwidths,
            extra={
                "options": options,
                "incumbents": incumbents,
                "keep_resolutions": planner.keep_resolutions,
            },
        )
        for index, problem in enumerate(problems)
        for budget in budgets
    }
    if any(row[2] is None for row in requests.values()):
        return None
    for index, problem in enumerate(problems):
        planner._shared_lanes[problem.digest] = requests[index, budgets[0]][1]
    return requests


def prepare_geometry(planner, problems, budgets, *, incumbents):
    bound = current()
    assert bound is not None
    control = bound.control
    control.agree("sweep/symmetric", bound.specification.symmetric_planning)
    if not bound.specification.symmetric_planning:
        return {}
    # Several ordering labels can lower to the same program. Keep those labels,
    # while searching the identical problem only once per budget.
    unique = {}
    aliases = [unique.setdefault(problem.digest, len(unique)) for problem in problems]
    peers = control.exchange("sweep/ordering_aliases", aliases)
    if any(value != peers[0] for value in peers):
        print(
            "ShadowSpill: symmetric planning fallback (ordering equivalence).",
            flush=True,
        )
        return {}
    problems = tuple({problem.digest: problem for problem in problems}.values())
    budgets = tuple(sorted(budgets))
    requests = _verified_requests(planner, problems, budgets, incumbents=incumbents)
    if requests is None:
        return {}
    sweep = _SharedSweep(
        planner=planner,
        control=control,
        problems=problems,
        budgets=budgets,
        aliases=aliases,
        requests=requests,
        incumbents=incumbents,
        owners={
            index: control.members[index % len(control.members)]
            for index in range(len(problems))
        },
    )
    local = control.run("sweep/shared_search", sweep.search_owned)
    peers = control.exchange("sweep/shared_results", local)
    return control.run("sweep/shared_admission", partial(sweep.admit, peers))


@dataclass
class _SharedSweep:
    """One verified geometry's ownership, search results, and local admissions."""

    planner: _Planner
    control: Control
    problems: tuple[ShadowSpillPlanningProblem, ...]
    budgets: tuple[tuple[int, int], ...]
    aliases: list[int]
    requests: dict
    incumbents: bool
    owners: dict[int, int]
    held: dict = field(default_factory=dict, init=False)

    def search_owned(self):
        results = []
        for index in range(len(self.problems)):
            if self.owners[index] != self.control.rank:
                continue
            carried = None
            for budget in self.budgets:
                self.control.check_failure()
                started = time.perf_counter()
                problem, transfer, _, _ = self.requests[index, budget]
                local = _Planner(
                    transfer,
                    self.planner.search_options,
                    self.planner.artifact_store,
                    self.planner.plan_store,
                    self.planner.plan_store_mode,
                    self.planner.verbose,
                    self.planner.keep_resolutions,
                )
                print(
                    f"ShadowSpill rank {self.control.rank}: "
                    f"search ordering {index + 1}/{len(self.problems)}, "
                    f"execution={budget[0]} spill={budget[1]} bytes",
                    flush=True,
                )
                row = {"ordering": index, "budget": budget}
                try:
                    answer = local.answer(
                        problem, *budget, carried if self.incumbents else None
                    )
                except _EXHAUSTED as error:
                    row.update(error=str(error), status="search_exhausted")
                except _INFEASIBLE as error:
                    row.update(error=str(error), status="infeasible")
                except AdmissionError as error:
                    row.update(error=str(error), status="rejected")
                else:
                    plan = answer.plan or local.plan(problem, *budget)
                    self.held[index, budget] = plan
                    row.update(
                        plan=pack(plan), inherited=answer.answered_with_incumbent
                    )
                    if carried is None or answer.makespan_ns < carried.makespan_ns:
                        carried = _Carried(*budget, answer.makespan_ns, plan)
                row["search_seconds"] = time.perf_counter() - started
                results.append(row)
        return results

    def admit(self, peers):
        store = ArtifactStore.resolve(
            self.planner.artifact_store,
            plan_store=self.planner.plan_store,
            plan_store_mode=self.planner.plan_store_mode,
        )
        answers = {}
        for owner, rows in zip(self.control.members, peers, strict=True):
            for row in rows:
                index, budget = row["ordering"], tuple(row["budget"])
                if (
                    index not in self.owners
                    or self.owners[index] != owner
                    or (index, budget) not in self.requests
                ):
                    raise ValueError("invalid shared sweep ownership")
                key = (self.problems[index].digest, *budget)
                if key in answers:
                    raise ValueError("duplicate shared sweep result")
                answers[key] = (
                    _refusal(row)
                    if "error" in row
                    else self.receive_result(index, budget, row, store)
                )
        if len(answers) != len(self.requests):
            raise ValueError("shared sweep results are incomplete")
        return answers

    def receive_result(self, index, budget, row, store):
        problem, transfer, shared, evidence = self.requests[index, budget]
        assert shared is not None
        plan = self.held.get((index, budget))
        if plan is None:
            plan = received_plan(shared, row["plan"], problem, budget, transfer)
        record_decision(
            store,
            {
                "version": 1,
                "members": self.control.members,
                "rank": self.control.rank,
                "local_program": problem.program.digest,
                "planning": evidence,
                "ordering": index,
                "equivalent_orderings": [
                    position
                    for position, alias in enumerate(self.aliases)
                    if alias == index
                ],
                "execution_budget_bytes": budget[0],
                "spill_budget_bytes": budget[1],
                "searched_by_rank": self.owners[index],
                "shared_plan": record_plan(store, plan),
            },
        )
        return _Answer(
            plan.simulation.makespan_ns,
            summarize_selected_plan(plan.result),
            graph_pair_outcomes(plan.result),
            row["inherited"],
            plan,
            row["search_seconds"],
        )


def _refusal(row):
    if row["status"] == "infeasible":
        error = PlanInfeasibleError(row["error"], kind="shared_infeasible")
    elif row["status"] == "search_exhausted":
        error = PlanSearchExhaustedError(row["error"])
    else:
        error = AdmissionError(row["error"])
    error.search_seconds = row["search_seconds"]
    return error
