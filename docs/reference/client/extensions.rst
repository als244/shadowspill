Writing extensions
==================

An extension package adds components to the catalog: models, objectives and
optimizers the service builds by name. These are the records a component
declares about itself: its inputs, per-call controls, outputs, loss
participation and source. Import them from ``shadowspill_client.extensions``.

.. autoclass:: shadowspill_client.extensions.InputSchema

.. autoclass:: shadowspill_client.extensions.TensorField

.. autoclass:: shadowspill_client.extensions.Capacity

.. autoclass:: shadowspill_client.extensions.ControlSchema

.. autoclass:: shadowspill_client.extensions.ScalarControl

.. autoclass:: shadowspill_client.extensions.OutputSet

.. autoclass:: shadowspill_client.extensions.TensorOutput

.. autoclass:: shadowspill_client.extensions.Participation

.. autoclass:: shadowspill_client.extensions.ComponentSource
