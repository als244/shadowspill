Frontend
========

The PyTorch frontend: the controls that shape what a builder captures, and the
runtime opened for PyTorch. Each control is a separate decision:

- ``partition`` (`PartitionPolicy`, `IsolateOperations`) sets the
  checkpoint-stage boundaries;
- ``chunking`` (`ChunkPolicy`, `ChunkRegions`, `AxisChunks`) runs selected
  regions of a stage in slices along one axis;
- ``graph_pair_policy`` (`GraphPairPolicy`) sets what each stage's save and
  recompute variants keep for backward: each variant's `RetentionPolicy`
  threshold, rules and default, with the flop formulas custom operators
  register, decide its saved-state cut;
- ``gradient_destination_rules`` (`GradientDestinationRule`) let an opaque
  operator add into existing gradient storage;
- `ChunkRegions.lifetimes` (`ExecutionLifetimes`) moves, compacts or replays
  operations inside a task.

Changing a control changes the tasks, their memory and their time, so each
candidate is built and profiled; planning then chooses among the measured
variants and where their values live. A value a variant keeps for backward may
still be spilled.

.. autoclass:: shadowspill.pytorch.Runtime

.. autoclass:: shadowspill.frontend.RuntimeFrontend

.. autoclass:: shadowspill.pytorch.InvocationResult
   :exclude-members: wait

.. autoclass:: shadowspill.pytorch.ObjectiveResult

.. autoclass:: shadowspill.pytorch.TensorSpec

.. autoclass:: shadowspill.pytorch.PartitionPolicy

.. autodata:: shadowspill.pytorch.partition.policy.PartitionSpec

.. autoclass:: shadowspill.pytorch.IsolateOperations

.. autoclass:: shadowspill.pytorch.GraphPairPolicy

.. autoclass:: shadowspill.pytorch.RetentionPolicy

.. autoclass:: shadowspill.pytorch.RetentionRule

.. autoclass:: shadowspill.pytorch.OperatorClass

.. autoclass:: shadowspill.pytorch.ChunkPolicy

.. autoclass:: shadowspill.pytorch.StageInfo

.. autoclass:: shadowspill.pytorch.capture.TensorGeometry

.. autoclass:: shadowspill.pytorch.AxisChunks

.. autoclass:: shadowspill.pytorch.ChunkAxis

.. autoclass:: shadowspill.pytorch.AxisFrom

.. autoclass:: shadowspill.pytorch.ChunkLocal

.. autoclass:: shadowspill.pytorch.OperatorRule

.. autoclass:: shadowspill.pytorch.ChunkRegions

.. autoclass:: shadowspill.pytorch.ExecutionLifetimes

.. autoclass:: shadowspill.pytorch.LifetimeRule

.. autoclass:: shadowspill.pytorch.LifetimeSelection

.. autoclass:: shadowspill.pytorch.LifetimeOperation

.. autoclass:: shadowspill.pytorch.ReadonlyOperator

.. autoclass:: shadowspill.pytorch.LayoutIndependentOperator

.. autoclass:: shadowspill.pytorch.GradientDestinationRule

.. autoclass:: shadowspill.pytorch.GradientDestinationContext

.. autoclass:: shadowspill.pytorch.GradientDestination

.. autoclass:: shadowspill.pytorch.GradientDestinationMapping

.. autoclass:: shadowspill.pytorch.DestinationView
