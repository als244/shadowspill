Parallelism
===========

`Parallelism.to_dict <shadowspill.Parallelism.to_dict>` and
`Parallelism.from_dict <shadowspill.Parallelism.from_dict>` use the plain
form a client passes as a run's ``parallelism`` setting, which
:ref:`the parallelism declaration <parallelism-declaration>` describes.

Declare how a model's parameters, gradients and optimizer state are owned
across ranks (`Parallelism`), and how each batch's inputs, loss and outputs
divide among them (`InputPlacement`). Gradients are summed across the ranks
that hold copies of a parameter, never averaged: each rank's objective returns
its share of the whole update, such as its loss sum divided by the update's
global count.

.. autoclass:: shadowspill.Parallelism

.. autoclass:: shadowspill.Replicated

.. autoclass:: shadowspill.Sharded

.. autoclass:: shadowspill.ParameterStorage

.. autoclass:: shadowspill.GroupRef

.. autoclass:: shadowspill.parallelism.InputPlacement

.. autoclass:: shadowspill.parallelism.InputFieldPlacement

.. autoclass:: shadowspill.parallelism.OutputPlacement
