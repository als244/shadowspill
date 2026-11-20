Memory search
=============

The search that places a program's objects within device and spill budgets.
It has two layers:

- `~shadowspill.planning.memory_search.plan_program` answers a self-contained
  `ShadowSpillPlanningProblem` with an `AnnotatedProgramPlan`, and keeps each
  answer in the plan store, keyed by the question.
- A `SearchAlgorithm`, which it runs, answers one program, its residency
  and a machine with a `ProgramPlanResult`, and stores nothing. The
  search is replaceable: a subclass passed in `SearchOptions` runs in place of
  the shipped `pressurefit`, and the toolkit functions below serve any search.

.. autofunction:: shadowspill.planning.memory_search.plan_program

.. autoclass:: shadowspill.planning.memory_search.program_inputs.ShadowSpillPlanningProblem

.. autoclass:: shadowspill.planning.memory_search.program_inputs.TransferBandwidths

.. autoclass:: shadowspill.planning.memory_search.annotated_plan.AnnotatedProgramPlan

.. autoclass:: shadowspill.planning.memory_search.SearchOptions

.. autoclass:: shadowspill.planning.memory_search.GenericPlanningOptions

.. autoclass:: shadowspill.planning.memory_search.SearchAlgorithm
   :special-members: __call__

.. autodata:: shadowspill.planning.memory_search.algorithms.pressurefit.pressurefit
   :no-value:

.. autoclass:: shadowspill.planning.memory_search.algorithms.pressurefit.PressureFit

.. autoclass:: shadowspill.planning.memory_search.algorithms.pressurefit.PressureFitOptions

.. autoclass:: shadowspill.planning.memory_search.OptionRecord

.. autoclass:: shadowspill.planning.memory_search.ProgramPlanResult

.. autoclass:: shadowspill.planning.memory_search.result.ResolutionPlan

.. autoclass:: shadowspill.planning.memory_search.ResidentSlice

.. autoclass:: shadowspill.planning.memory_search.AdmissionFacts

.. autoclass:: shadowspill.planning.memory_search.TaskAdmissionSpec

.. autoclass:: shadowspill.planning.memory_search.TaskAllocationStep

.. autoclass:: shadowspill.planning.memory_search.TaskAllocationStepKind

.. autoclass:: shadowspill.planning.memory_search.StorageHandoff

.. autoclass:: shadowspill.planning.memory_search.admission.layout.FixedPhysicalLayout

.. autoclass:: shadowspill.planning.memory_search.admission.layout.model.FixedLayoutPlacements

.. autoclass:: shadowspill.planning.memory_search.admission.layout.FixedLayoutPlacement

.. autoclass:: shadowspill.planning.memory_search.admission.layout.model.FixedLayoutReuses

.. autoclass:: shadowspill.planning.memory_search.admission.layout.FixedLayoutReuse

.. autoclass:: shadowspill.planning.memory_search.admission.layout.model.FixedLayoutTaskAllocations

.. autoclass:: shadowspill.planning.memory_search.admission.layout.model.LeaseLifetime

.. autoexception:: shadowspill.planning.memory_search.admission.layout.FixedLayoutInfeasibleError

.. autofunction:: shadowspill.planning.memory_search.validate_search_inputs

.. autofunction:: shadowspill.planning.memory_search.validate_schedule_feasibility

.. autofunction:: shadowspill.planning.memory_search.answer_no_worse_than

.. autoclass:: shadowspill.planning.memory_search.CostedAlternatives

.. autoclass:: shadowspill.planning.memory_search.toolkit.resolution.CostedGroup

.. autoclass:: shadowspill.planning.memory_search.toolkit.resolution.CostedOption

.. autodata:: shadowspill.planning.memory_search.toolkit.resolution.Resolution

.. autofunction:: shadowspill.planning.memory_search.resolutions

.. autodata:: shadowspill.planning.memory_search.toolkit.resolution.DEFAULT_RESOLUTION_OPTIONS

.. autodata:: shadowspill.planning.memory_search.toolkit.resolution.NAMED_RESOLUTION_OPTIONS

.. autofunction:: shadowspill.planning.memory_search.named_resolution_options

.. autofunction:: shadowspill.planning.memory_search.validate_resolution_options

.. autofunction:: shadowspill.planning.memory_search.validate_tail_save_counts

.. autoclass:: shadowspill.planning.memory_search.PlanningDiagnostics

.. autoclass:: shadowspill.planning.memory_search.diagnostics.plan_summary.PlanSummary

.. autoclass:: shadowspill.planning.memory_search.GraphPairOutcome

.. autoclass:: shadowspill.planning.memory_search.ResolvedProgramDiagnostics

.. autoclass:: shadowspill.planning.memory_search.CandidateDiagnostic

.. autoclass:: shadowspill.planning.memory_search.diagnostics.resolved_programs.IncumbentDiagnostic

.. autoclass:: shadowspill.planning.memory_search.TaskAlternativeChoiceDiagnostic

.. autoclass:: shadowspill.planning.memory_search.PlanningRepairDiagnostics

.. autoclass:: shadowspill.planning.memory_search.PlanningWorkDiagnostics

.. autoclass:: shadowspill.planning.memory_search.PlanningSectionTiming

.. autoclass:: shadowspill.planning.memory_search.ReductionStep
