Diagnostics and errors
======================

Reports of planned work, what a traced run measured, and the errors the
phases raise. A failure in a live runtime raises `RuntimeExecutionError`
instead.

.. autofunction:: shadowspill.write_report

.. autoclass:: shadowspill.diagnostics.StepDiagnostics

.. autoclass:: shadowspill.diagnostics.StepTimingSummary

.. autoclass:: shadowspill.diagnostics.Timelines

.. autoexception:: shadowspill.errors.PlanningError
   :show-inheritance:

.. autoexception:: shadowspill.errors.CaptureError
   :show-inheritance:

.. autoexception:: shadowspill.errors.TaskPhaseError
   :show-inheritance:

.. autoexception:: shadowspill.errors.CompilationError
   :show-inheritance:

.. autoexception:: shadowspill.errors.ProfilingError
   :show-inheritance:

.. autoexception:: shadowspill.errors.AdmissionError
   :show-inheritance:

.. autoexception:: shadowspill.errors.PlanInfeasibleError
   :show-inheritance:

.. autoexception:: shadowspill.errors.PlanSearchExhaustedError
   :show-inheritance:

.. autoexception:: shadowspill.errors.ObjectiveError
   :show-inheritance:

.. autoexception:: shadowspill.errors.InputGuardError
   :show-inheritance:
