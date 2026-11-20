Programs and schedules
======================

The framework-neutral program a frontend lowers a model into, and the
schedules planned for it. Nothing here imports a framework, so a saved program
can be planned, simulated and admitted with no model and no device.

- A `ShadowSpillProgram` holds the objects (`ObjectSpec`), the storage they
  occupy (`AliasGroupSpec`), the tasks that read and write them (`TaskSpec`,
  costed by `~shadowspill.ir.TaskProfile`), and the alternative task sequences
  planning chooses between (`TaskAlternativeGroup`).
- A `MemorySchedule` says where each storage starts, which `MemoryAction`
  moves it at each task boundary, and where it ends; an `ExecutionPlan` binds
  a schedule to its selections, entrypoints and admitted memory.
- `index_program`, `index_memory_schedule` and `index_execution_plan` flatten
  a value into arrays once, for a caller that plans or simulates it
  repeatedly.

.. autoclass:: shadowspill.ir.ShadowSpillProgram

.. autoclass:: shadowspill.ir.DeviceSpec

.. autoclass:: shadowspill.ir.ObjectSpec

.. autoclass:: shadowspill.ir.ObjectRole

.. autoclass:: shadowspill.ir.Persistence

.. autoclass:: shadowspill.ir.AliasGroupSpec

.. autoclass:: shadowspill.ir.MutationSpec

.. autoclass:: shadowspill.ir.TransferSpec

.. autoclass:: shadowspill.ir.TransferPhase

.. autoclass:: shadowspill.ir.SharedResidencyPolicy

.. autoclass:: shadowspill.ir.ResourceSpec

.. autoclass:: shadowspill.ir.ResourceKind

.. autoclass:: shadowspill.ir.TaskSpec

.. autoclass:: shadowspill.ir.TaskProfile

.. autoclass:: shadowspill.ir.TaskResourcePhase

.. autoclass:: shadowspill.ir.TaskOverlapCost

.. autoclass:: shadowspill.ir.TaskAlternativeGroup

.. autoclass:: shadowspill.ir.TaskAlternativeOption

.. autoclass:: shadowspill.ir.TaskAlternativeChoice

.. autoclass:: shadowspill.ir.MemorySchedule

.. autoclass:: shadowspill.ir.MemoryAction

.. autoclass:: shadowspill.ir.MemoryActionKind

.. autoclass:: shadowspill.ir.MemoryLocation

.. autoclass:: shadowspill.ir.ResidencySpec

.. autoclass:: shadowspill.ir.ExecutionPlan

.. autoclass:: shadowspill.ir.EntrypointSpec

.. autoclass:: shadowspill.ir.PhysicalAdmission

.. autoclass:: shadowspill.ir.PlanPrediction

.. autofunction:: shadowspill.ir.shared_residency_footprint

.. autoclass:: shadowspill.ir.SharedResidencyFootprint

.. autofunction:: shadowspill.ir.index_program

.. autoclass:: shadowspill.ir.IndexedProgram

.. autofunction:: shadowspill.ir.index_memory_schedule

.. autoclass:: shadowspill.ir.IndexedMemorySchedule

.. autofunction:: shadowspill.ir.index_execution_plan

.. autoclass:: shadowspill.ir.IndexedExecutionPlan

.. autoexception:: shadowspill.ir.ValidationError
