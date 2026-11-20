Service
=======

Run the training service that clients connect to with
`shadowspill_client.ServiceClient`. A `Server` serves one `Allocation`;
`ServerLimits` bounds the memory, storage and concurrency of its runs and
workers.

.. autoclass:: shadowspill.server.Server

.. autoclass:: shadowspill.server.Allocation

.. autoclass:: shadowspill.server.ServerLimits

.. autoclass:: shadowspill.server.startup.configuration.LocalCreationLimits

.. autoclass:: shadowspill.orchestration.capture.files.CollectorLimits

.. autoclass:: shadowspill.training.worker.resources.WorkerHost

.. autoclass:: shadowspill.training.worker.protocol.WorkerStartup

.. autoclass:: shadowspill.training.worker.protocol.WorkerScope

.. autoclass:: shadowspill.server.authentication.BearerTokens

.. autoclass:: shadowspill.orchestration.nodes.authentication.NodeTokens

.. autoclass:: shadowspill.orchestration.nodes.authentication.NodeAccess
