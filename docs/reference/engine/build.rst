Build
=====

Capture a model and representative calls as a `Program`. A builder
partitions the model's computation into checkpoint stages, lowers each to
tasks and finalizes their code on a representative device: `build_step`
captures a complete training update, `build_forward_backward` the objective
and backward without an optimizer, and `build_forward` one forward call.

.. autofunction:: shadowspill.build_step

.. autofunction:: shadowspill.build_forward_backward

.. autofunction:: shadowspill.build_forward

.. autoclass:: shadowspill.Program
   :show-inheritance:

.. autoclass:: shadowspill.artifacts.TaskDefinition

.. autoclass:: shadowspill.artifacts.archive.Artifact

.. autoclass:: shadowspill.OperatorPreparation

.. autoclass:: shadowspill.Resource
