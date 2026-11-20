Planning
========

Choose how a `ProfiledProgram` runs within `MemoryBudgets`, on the CPU:
`~shadowspill.plan_program` plans one Program, and `sweep_programs` several
candidates at several budgets. The order a step walks its examples
(`StepDataOrdering`) is a planning choice; their shapes are fixed at build. `optimizer_program`, `backward_program` and `instantiate_examples`
derive other Programs from a captured one without rebuilding it, and
`PlanningInputs` replans the measurements a service preparation exports.

.. autofunction:: shadowspill.plan_program

.. autoclass:: shadowspill.MemoryBudgets

.. autoclass:: shadowspill.Plan
   :show-inheritance:

.. autofunction:: shadowspill.sweep_programs

.. autoclass:: shadowspill.SweepResult
   :show-inheritance:

.. autoclass:: shadowspill.artifacts.sweep.SweepCase

.. autofunction:: shadowspill.optimizer_program

.. autofunction:: shadowspill.backward_program

.. autofunction:: shadowspill.instantiate_examples

.. autoclass:: shadowspill.PlanningInputs

.. autoclass:: shadowspill.training.preparation.planning.requests.PlanningRequest

.. autoclass:: shadowspill.training.preparation.capture.templates.TemplateUse

.. autoclass:: shadowspill.training.preparation.offline.plans.LocalPlans
   :show-inheritance:

.. autoclass:: shadowspill.training.preparation.offline.sweep.PlanningSweep

.. autoclass:: shadowspill.training.preparation.offline.sweep.SweepRow
