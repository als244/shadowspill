Simulator
=========

Price a schedule without running it: `simulate` replays it through the
compiled simulator and returns a `SimulationResult` with every task's and
transfer's interval and the memory peaks.

.. autofunction:: shadowspill.simulator.simulate

.. autoclass:: shadowspill.simulator.SimulationConfig

.. autoclass:: shadowspill.simulator.DeviceSimulationConfig

.. autoclass:: shadowspill.simulator.SimulationAdmission

.. autoclass:: shadowspill.simulator.SimulationResult

.. autoexception:: shadowspill.simulator.SimulationInfeasibleError

.. autoclass:: shadowspill.simulator.TaskInterval

.. autoclass:: shadowspill.simulator.TransferInterval

.. autoclass:: shadowspill.simulator.TransferDirection

.. autoclass:: shadowspill.simulator.MemorySnapshot

.. autoclass:: shadowspill.simulator.DeviceMemoryPeak

.. autoclass:: shadowspill.simulator.CapacityViolation

.. autoclass:: shadowspill.simulator.ActionPhysicalDelta

.. autoclass:: shadowspill.simulator.TaskPhysicalDelta

.. autoclass:: shadowspill.simulator.MemoryReuseDependency
