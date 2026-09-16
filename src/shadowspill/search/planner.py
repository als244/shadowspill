"""The rule one point is answered by: the summary, the plan to beat, the search."""

from dataclasses import dataclass
from os import PathLike

from shadowspill.planner import (
    SearchOptions,
    plan_program,
    summarize_plan,
)
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import (
    INCUMBENT_CANDIDATE_ID,
    GraphPairOutcome,
    graph_pair_outcomes,
)
from shadowspill.planner.diagnostics.plan import (
    PlanSummary,
    summarize_selected_plan,
)
from shadowspill.planner.program_inputs import (
    ShadowSpillPlanningProblem,
    TransferBandwidths,
)
from shadowspill.search.refusals import _REFUSED
from shadowspill.store import StoreMode

# a point the planner refuses, for whatever reason it gives, is recorded and
# the sweep goes on; ProblemPreparationError is one such RuntimeError
_REJECTED = (RuntimeError,)


@dataclass(slots=True)
class _Carried:
    """The best plan found so far for one program, handed to every larger
    budget as the plan to beat, with the budget it was found at, which a
    point that answers with it records.

    The plan itself is held when this search planned it, and read from the
    store the first time a search needs it otherwise, so a point a summary
    answered costs nothing until a later point has to beat it.
    """

    execution_budget_bytes: int
    spill_budget_bytes: int
    makespan_ns: int
    plan: AnnotatedProgramPlan | None = None


@dataclass(frozen=True, slots=True)
class _Best:
    """The fastest plan seen under one budget pair, across every geometry
    and ordering: what a run of the winner starts from."""

    makespan_ns: int
    problem: ShadowSpillPlanningProblem
    #: `None` when a summary answered the point; the store holds the plan.
    plan: AnnotatedProgramPlan | None


@dataclass(frozen=True, slots=True)
class _Answer:
    """What one point's summary read, or its search, said."""

    makespan_ns: int
    summary: PlanSummary
    outcomes: tuple[GraphPairOutcome, ...]
    answered_with_incumbent: bool
    #: The plan when this point planned it; `None` when a summary answered.
    plan: AnnotatedProgramPlan | None


@dataclass(frozen=True, slots=True)
class _Planner:
    """The one way every point of a search asks the planner and its store.

    Every point is asked under the same lanes, options and store arguments.
    Holding them in one place keeps the summary read, the search and the
    read-back of a winner asking one question, so a summary stands for
    exactly the plan :func:`plan_program` would return.
    """

    transfer_bandwidths: TransferBandwidths | None
    search_options: SearchOptions | None
    artifact_store: str | PathLike[str] | None
    plan_store: str | PathLike[str] | None
    plan_store_mode: StoreMode
    verbose: bool

    def plan(
        self,
        problem: ShadowSpillPlanningProblem,
        execution_budget: int,
        spill_budget: int,
        incumbent: AnnotatedProgramPlan | None = None,
    ) -> AnnotatedProgramPlan:
        """Plan one point, or read its plan back: the one `plan_program` call."""

        return plan_program(
            problem,
            execution_budget=execution_budget,
            spill_budget=spill_budget,
            transfer_bandwidths=self.transfer_bandwidths,
            search_options=self.search_options,
            incumbent=incumbent,
            artifact_store=self.artifact_store,
            plan_store=self.plan_store,
            plan_store_mode=self.plan_store_mode,
            verbose=self.verbose,
        )

    def carried_plan(
        self, problem: ShadowSpillPlanningProblem, carried: _Carried
    ) -> AnnotatedProgramPlan:
        """The plan to beat itself, read from the store the first time."""

        if carried.plan is None:
            carried.plan = self.plan(
                problem, carried.execution_budget_bytes, carried.spill_budget_bytes
            )
        return carried.plan

    def answer(
        self,
        problem: ShadowSpillPlanningProblem,
        execution_budget: int,
        spill_budget: int,
        carried: _Carried | None,
    ) -> _Answer:
        """Answer one point from its summary while the stored answer stands,
        else plan it.

        The stored answer stands unless the plan in hand claims to beat it,
        which is the store's own rule for a plan it is handed: then the
        search runs with that plan and keeps the better. A refusal the store
        recorded is final without a plan in hand, and a question for the
        search with one.
        """

        try:
            stored = summarize_plan(
                problem,
                execution_budget=execution_budget,
                spill_budget=spill_budget,
                transfer_bandwidths=self.transfer_bandwidths,
                search_options=self.search_options,
                artifact_store=self.artifact_store,
                plan_store=self.plan_store,
                plan_store_mode=self.plan_store_mode,
            )
        except _REFUSED:
            if carried is None:
                raise
            stored = None
        if stored is not None and (
            carried is None or carried.makespan_ns >= stored.makespan_ns
        ):
            return _Answer(
                stored.makespan_ns,
                stored.summary,
                stored.graph_pair_outcomes,
                stored.answered_with_incumbent,
                None,
            )
        incumbent = None if carried is None else self.carried_plan(problem, carried)
        plan = self.plan(problem, execution_budget, spill_budget, incumbent)
        return _Answer(
            plan.simulation.makespan_ns,
            summarize_selected_plan(plan.result),
            graph_pair_outcomes(plan.result),
            plan.result.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID,
            plan,
        )
