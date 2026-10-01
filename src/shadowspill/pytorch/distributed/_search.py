"""PyTorch preparation wrapper around the unchanged neutral sweep planner."""

from __future__ import annotations

from dataclasses import replace

from shadowspill.ir import ShadowSpillProgram
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import graph_pair_outcomes
from shadowspill.planner.diagnostics.plan import summarize_selected_plan
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.search.planner import _Answer, _Carried, _Planner
from shadowspill.store import ArtifactStore

from . import current
from ._selection import (
    certify_restored,
    choose,
    fixed_admission,
    record_decision,
    restore_result,
)


class DistributedPlanner(_Planner):
    def plan(
        self,
        problem: ShadowSpillPlanningProblem,
        execution_budget: int,
        spill_budget: int,
        incumbent: AnnotatedProgramPlan | None = None,
    ) -> AnnotatedProgramPlan:
        def attempt(
            fixed: ShadowSpillProgram, carried: ProgramPlanResult | None
        ) -> AnnotatedProgramPlan:
            fixed_problem = replace(
                problem,
                program=fixed,
                admission_facts=fixed_admission(problem.admission_facts, fixed),
            )
            # Reuse the corresponding local incumbent, with its original public
            # source problem restored in the result returned below.
            fixed_incumbent = None
            if carried is not None:
                assert incumbent is not None
                fixed_incumbent = replace(
                    incumbent, program=fixed_problem, result=carried, _digest_cache=[]
                )
            return super(DistributedPlanner, self).plan(
                fixed_problem, execution_budget, spill_budget, fixed_incumbent
            )

        local, key, successful, decisions = choose(
            problem.program,
            attempt,
            search_options=self.search_options,
            incumbent=None if incumbent is None else incumbent.result,
        )
        result = restore_result(
            local,
            problem.program,
            key,
            successful,
            keep_resolutions=self.keep_resolutions,
        )
        store = ArtifactStore.resolve(
            self.artifact_store,
            plan_store=self.plan_store,
            plan_store_mode=self.plan_store_mode,
        )
        facts, admission = certify_restored(
            result,
            problem.admission_facts,
            local.effective_facts,
            scratch_reserve_bytes=local.fixed_layout.scratch_reserve_bytes,
        )
        record_decision(store, decisions)
        return replace(
            local,
            program=problem,
            result=result,
            effective_facts=facts,
            fixed_layout=admission.layout,
            simulation_admission=admission.simulator_input,
            simulation=admission.simulation,
            _digest_cache=[],
        )

    def answer(
        self,
        problem: ShadowSpillPlanningProblem,
        execution_budget: int,
        spill_budget: int,
        carried: _Carried | None,
    ) -> _Answer:
        incumbent = None if carried is None else self.carried_plan(problem, carried)
        plan = self.plan(problem, execution_budget, spill_budget, incumbent)
        bound = current()
        assert bound is not None
        score = max(
            bound.control.exchange(
                "sweep/local_prediction", plan.simulation.makespan_ns
            )
        )
        return _Answer(
            score,
            summarize_selected_plan(plan.result),
            graph_pair_outcomes(plan.result),
            False,
            plan,
        )
