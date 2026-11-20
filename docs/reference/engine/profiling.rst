Profiling
=========

Measure what a `Program` costs on the hardware it will run on:
`measure_program` its tasks' time and memory, `measure_communication` its
transfers and collectives, `measure_overlap` their contention with its tasks,
and `measure_pool_memory` a spill pool's device memory. Each result is saved
and passed to planning, which measures nothing; `apply_profiles` attaches the
task profiles, with any overlap profiles merged in, on the CPU.

.. autofunction:: shadowspill.measure_program

.. autoclass:: shadowspill.ProfilingOptions

.. autoclass:: shadowspill.task.profiling.TimingWindow

.. autoclass:: shadowspill.TaskProfileCache

.. autoclass:: shadowspill.profiling.cache.CachedTask

.. autoclass:: shadowspill.Profiles
   :show-inheritance:

.. autoclass:: shadowspill.TaskProfile

.. autofunction:: shadowspill.apply_profiles

.. autoclass:: shadowspill.ProfiledProgram
   :show-inheritance:

.. autofunction:: shadowspill.measure_communication

.. autoclass:: shadowspill.CommunicationInfo
   :show-inheritance:

.. autofunction:: shadowspill.measure_overlap

.. autoclass:: shadowspill.OverlapProfileCache

.. autofunction:: shadowspill.measure_pool_memory

.. autoclass:: shadowspill.PoolMemoryProfile
   :show-inheritance:

.. autoexception:: shadowspill.profiling.pool_memory.PoolMemoryMeasurementError
