Training
========

Submit training operations and read their results.

.. autoclass:: shadowspill_client.TrainingClient
   :members: step, forward_backward, optim_step, forward, discard_gradients,
      save_checkpoint, load_checkpoint, export_weights, validate_batch,
      get_config, log_metrics, report, nsys, close

.. autoclass:: shadowspill_client.creation.TrainingFuture

.. autoclass:: shadowspill_client.results.OperationFuture

.. autoclass:: shadowspill_client._async.CallFuture

.. autoclass:: shadowspill_client.results.TrainingResult

.. autoclass:: shadowspill_client.protocol.results.OperationResult

.. autoclass:: shadowspill_client.protocol.results.Accumulation

.. autoclass:: shadowspill_client.protocol.results.ArtifactIssue

.. autoclass:: shadowspill_client.protocol.results.Acceptance

.. autoclass:: shadowspill_client.protocol.operations.OperationKey

.. autoclass:: shadowspill_client.protocol.requests.RequestIdentity

.. autoclass:: shadowspill_client.training.RequestQueries

.. autoclass:: shadowspill_client.protocol.results.RequestStatus

.. autoclass:: shadowspill_client.protocol.results.Failure

