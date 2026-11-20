Artifact store
==============

Optional caches of captured graphs, measurements and planner results, each
entry keyed by the inputs that affect it. A run says what it may do with each
of the store's trees as a `StoreMode`, which `StorePolicy` turns into what it
may read and write.

.. autoclass:: shadowspill.store.ArtifactStore

.. autoclass:: shadowspill.store.StorePolicy

.. autotype:: shadowspill.store.policy.StoreMode

.. autodata:: shadowspill.store.policy.STORE_MODES

.. autodata:: shadowspill.store.policy.CONTRIBUTE

.. autoclass:: shadowspill.store.PlanningArtifact

.. autoclass:: shadowspill.store.ArtifactRecorder

.. autofunction:: shadowspill.store.digest_directory
