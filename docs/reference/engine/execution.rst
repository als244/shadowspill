Execution and checkpoints
=========================

Run a Plan, and save and load its state. An `Executor` opens the process's
one runtime, which installs the device allocator for the life of the process:
a process runs one Executor, and after `Executor.close` or a failed run the
next needs a new process. The spill pool it opens (`SpillPool` and its kinds)
must match the one the Plan's transfers were measured with.

.. autoclass:: shadowspill.Executor

.. autoclass:: shadowspill.TrainingResult

.. autoclass:: shadowspill.TensorChunk

.. autofunction:: shadowspill.save_checkpoint

.. autofunction:: shadowspill.load_checkpoint

.. autofunction:: shadowspill.checkpoint.write_checkpoint

.. autofunction:: shadowspill.checkpoint.checkpoint_size

.. autofunction:: shadowspill.checkpoint.write_weights

.. autofunction:: shadowspill.checkpoint.weights_size

.. autoclass:: shadowspill.memory.SpillPool

.. autoclass:: shadowspill.memory.PinnedHostPool
   :show-inheritance:

.. autofunction:: shadowspill.memory.pinned_host

.. autoclass:: shadowspill.ssd.SSDPool
   :show-inheritance:

.. autofunction:: shadowspill.ssd.ssd

.. autoclass:: shadowspill.network.RemotePool
   :show-inheritance:

.. autofunction:: shadowspill.network.remote
