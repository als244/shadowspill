Steps and tasks
===============

A step's and a task's records, which planning, simulation, the runtime and the
diagnostics read in place of anything a framework produced.
`StepDataOrdering` is how a step walks its examples. The task records say
what a frontend captured, compiled and measured for one task: the roles of its
arguments (`TaskInputRole`), the storage it promises to use
(`TaskStorageContract`), the allocations its profiled runs made
(`TaskAllocationContract`), and its measurement (`TaskMeasurement`).

.. autoclass:: shadowspill.step.StepDataOrdering

.. autoclass:: shadowspill.task.TaskInputRole

.. autodata:: shadowspill.task.inputs.REPRESENTATIVE_VALUE_POLICY

.. autoclass:: shadowspill.task.storage.TaskStorageContract

.. autoclass:: shadowspill.task.storage.StorageRoot

.. autoclass:: shadowspill.task.storage.StorageRootKind

.. autoclass:: shadowspill.task.storage.OutputView

.. autoclass:: shadowspill.task.storage.MutationBinding

.. autoclass:: shadowspill.task.allocations.contract.TaskAllocationContract

.. autoclass:: shadowspill.task.allocations.contract.TaskAllocationContractStep

.. autoclass:: shadowspill.task.allocations.events.TaskAllocationEvent

.. autoclass:: shadowspill.task.allocations.events.TaskAllocationOperation

.. autoclass:: shadowspill.task.profiles.TaskMeasurement

.. autoclass:: shadowspill.task.profiles.TaskOutputInputBinding
