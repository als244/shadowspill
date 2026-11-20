Runtime
=======

The runtime a Plan executes on. A process opens one
`~shadowspill.runtime.Runtime`, through a framework's `RuntimeFrontend`,
before the framework makes any device allocation; `Executor` opens it in each
worker. It holds the pools ``shadowspill.memory`` configures, one `DevicePool`
and one or more spill pools, and the directed routes between them
(`TransferRoute`). The occupancy, timing and admission-replay functions here
serve tooling.

.. autoclass:: shadowspill.runtime.Runtime

.. autoclass:: shadowspill.runtime.core.PlanState

.. autoclass:: shadowspill.runtime.MemoryPool

.. autoclass:: shadowspill.runtime.RuntimeRoute

.. autoclass:: shadowspill.runtime.TransferCapabilities

.. autoclass:: shadowspill.runtime.TransferProfile

.. autoclass:: shadowspill.memory.DevicePool

.. autofunction:: shadowspill.memory.device

.. autoclass:: shadowspill.memory.TransferRoute

.. autofunction:: shadowspill.memory.transfer_route

.. autodata:: shadowspill.memory.MemoryPoolConfig

.. autoclass:: shadowspill.runtime.abi.statistics.MemoryPoolStatistics

.. autoclass:: shadowspill.runtime.occupancy.PoolAllocation

.. autoclass:: shadowspill.runtime.ObjectRef

.. autoclass:: shadowspill.runtime.plan.tracing.TraceCapacities

.. autoclass:: shadowspill.runtime.timing.Marker

.. autofunction:: shadowspill.runtime.timing.nanoseconds_between

.. autofunction:: shadowspill.runtime.timing.wait_for_stream

.. autofunction:: shadowspill.runtime.run_admission_replay

.. autoclass:: shadowspill.runtime.AdmissionReplayOperation

.. autoclass:: shadowspill.runtime.AdmissionReplayOperationKind

.. autoclass:: shadowspill.runtime.AdmissionReplayLeaseState

.. autoclass:: shadowspill.runtime.AdmissionReplayDecision

.. autoclass:: shadowspill.runtime.AdmissionReuseDependency

.. autoclass:: shadowspill.runtime.AdmissionReplayResult

.. autoexception:: shadowspill.runtime.admission_replay.PhysicalAdmissionError

.. autoexception:: shadowspill.runtime.RuntimeConfigurationError

.. autoexception:: shadowspill.runtime.RuntimeExecutionError

.. autoclass:: shadowspill.runtime.RuntimeFailureDiagnostics
