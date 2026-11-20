Parallelism
===========

How a run is spread over its workers.

.. autofunction:: shadowspill_client.parallelism.data_parallel

.. autofunction:: shadowspill_client.parallelism.expert_parallel

.. autodata:: shadowspill_client.parallelism.MIN_SHARD_BYTES

.. _parallelism-declaration:

The parallelism declaration
---------------------------

`data_parallel` and `expert_parallel` return a parallelism declaration: a dict
of plain values that a run takes as its ``parallelism`` setting. Any dict of
this form can be passed instead. The server refuses one that breaks a rule
below: it checks the form when it resolves the run's settings, and the rules
that involve the model's parameters when it builds the model. Without a
``parallelism`` setting, a run uses the model's, or else one rank.

A run has ``ranks_per_replica * num_replicas`` ranks, numbered from 0. A
replica is ``ranks_per_replica`` consecutive ranks that together hold one copy
of the model state: replica ``k`` is ranks ``k * ranks_per_replica`` to
``(k + 1) * ranks_per_replica - 1``. Every replica has the same layout, so
state is never divided across replicas: the ranks at the same position of
every replica hold copies of the same state.

.. list-table::
   :header-rows: 1
   :widths: 24 22 18 36

   * - Key
     - Type
     - When omitted
     - Meaning
   * - ``ranks_per_replica``
     - int
     - 1
     - Ranks in one replica. Positive.
   * - ``num_replicas``
     - int
     - 1
     - Replicas. Positive.
   * - ``groups``
     - mapping of str to list of list of int
     - ``{"all_ranks": [[0, ..., ranks_per_replica - 1]]}``, also when empty
     - Group collections by name; see `Groups`_.
   * - ``default_parameter_placement``
     - placement
     - required
     - How a parameter no selector matches is computed with; see
       `Placements`_.
   * - ``parameter_placements``
     - mapping of str to placement
     - required; may be ``{}``
     - Placements by selector; see `Selectors`_.
   * - ``default_parameter_storage``
     - storage
     - required
     - How a parameter no selector matches is stored; see `Storage`_.
   * - ``parameter_storage``
     - mapping of str to storage
     - required; may be ``{}``
     - Storage by selector.
   * - ``inputs``
     - input placement
     - refused for more than one rank
     - How each batch's inputs, loss and outputs divide among the ranks; see
       `Inputs`_. Omitted, the one rank receives every input field whole.

No other key is accepted.

Groups
~~~~~~

A group collection is a list of groups of ranks within one replica, numbered
from 0 to ``ranks_per_replica - 1``. Its groups are nonempty and of equal size,
and together hold each rank of the replica exactly once. Order matters: a
rank's position in its group is its rank within that group. The server repeats
each collection in every replica, adding ``k * ranks_per_replica`` to each rank
of replica ``k``: with two replicas of four ranks, ``[[0, 1], [2, 3]]`` stands
for ``[[0, 1], [2, 3], [4, 5], [6, 7]]``. Placements and storage name a
collection by its key; a given ``groups`` has ``"all_ranks"`` only if it
declares it.

Placements
~~~~~~~~~~

A placement says which part of a parameter each rank computes with.

``{"kind": "Replicated"}``
   Every rank computes with the whole parameter and holds a copy of it.

``{"kind": "Sharded", "group": str, "dim": int}``
   The parameter is split into equal contiguous slices along ``dim``, one for
   each position of the collection's groups: the rank at position ``i`` of a
   group computes with slice ``i``, and the ranks at position ``i`` of every
   group, in every replica, hold copies of it. The model defines the parameter
   at the slice's shape; the whole parameter has that shape with dimension
   ``dim`` multiplied by the group size. ``dim`` indexes the parameter's
   dimensions, negative counting from the last.

A placement record has exactly the keys shown. A placement declares ownership
only: it divides no model code, and the model's layers perform any
communication a sharded parameter needs.

Storage
~~~~~~~

A storage record says where a parameter's weights, gradients and optimizer
state, master weights included, are stored. Each key is optional; omitted or
``None``, it is unset. No other key is accepted.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Type
     - Meaning
   * - ``weights``, ``gradients``, ``optimizer``
     - ``"sharded"`` or ``"replicated"``
     - The component's mode. ``"sharded"``: each rank of a storage group
       stores an equal part of the slice its ranks compute with, the whole
       parameter when ``Replicated``. ``"replicated"``: each rank stores all
       of that slice.
   * - ``group``
     - str
     - The collection the components are stored over.
   * - ``weight_group``, ``gradient_group``, ``optimizer_group``
     - str
     - The collection for one component, in place of ``group``.
   * - ``min_shard_bytes``
     - int
     - Weight storage size, in bytes, below which a parameter keeps every
       component whole on every rank. Nonnegative; zero, or unset in both the
       record and the default, shards every parameter.

- ``default_parameter_storage`` sets every component's mode, and a collection
  for every component through ``group`` or the component's own key.
- A record in ``parameter_storage`` takes each unset key from
  ``default_parameter_storage``, never from another record. A component is
  stored over the record's collection for that component, else the record's
  ``group``, else the default's collection for that component, else the
  default's ``group``.
- A storage group of a sharded component holds only ranks that compute with
  the same slice: any collection suits a ``Replicated`` parameter, while for a
  ``Sharded`` one each storage group may hold only ranks at one position of the
  placement's groups. A component stored whole, as described next, has no
  storage group to check.
- A component is stored whole on every rank when its mode is ``"replicated"``,
  when the parameter's weight storage is under ``min_shard_bytes``, and always
  for a buffer, which has no gradient or optimizer state.
- When the run has an optimizer, a trainable parameter whose weights or
  gradients are sharded over more than one rank must have its optimizer state
  sharded over the same groups.

Selectors
~~~~~~~~~

The keys of ``parameter_placements`` and ``parameter_storage`` are selectors: a
registered parameter or buffer name, such as ``"layers.0.mlp.weight"``, or a
pattern matched case-sensitively with shell wildcards, where ``*`` matches any
characters, dots included, ``?`` one character and ``[...]`` one of a set.

- Each selector is nonempty and matches at least one registered name.
- A selector equal to one of a parameter's names wins over every pattern;
  otherwise every pattern that matches applies.
- When several selectors apply to one parameter, including through the names
  of a parameter registered under more than one, they must declare the same
  value: no selector outranks another by its order or by being more specific.

Inputs
~~~~~~

``inputs`` divides each batch among the ranks. Unlike a group collection, it
numbers ranks across the whole run.

.. list-table::
   :header-rows: 1
   :widths: 14 24 16 46

   * - Key
     - Type
     - When omitted
     - Meaning
   * - ``version``
     - int
     - required
     - 1.
   * - ``partitions``
     - ``{"name": str, "groups": list of list of int}``
     - required
     - The input partitions, one group each, in partition order, under any
       nonempty ``name``. Each partition receives its own part of every batch
       and its members cooperate on it; its first member is its leader. The
       groups are nonempty and of equal size, and together hold every rank of
       the run exactly once.
   * - ``fields``
     - list of field records
     - required; may be ``[]``
     - How each input field reaches a partition's members, at most one record
       per path. When nonempty, it needs a record for every input field. Empty,
       every field is delivered whole, which the server refuses when a
       partition has more than one member.
   * - ``loss``
     - ``"leader"`` or ``"partitioned"``
     - ``"leader"``
     - Whose loss counts. ``"leader"``: each partition leader's, for a loss
       every member computes in full. ``"partitioned"``: every member's,
       summed, for a loss each member computes from its own part. Neither
       divides by a count: the objective computes the share it declares.
   * - ``outputs``
     - mapping of str to output record
     - required; may be ``{}``
     - How each named output is assembled from a partition's members. An
       output may go without a record only when every partition has one
       member.

A field record:

.. list-table::
   :header-rows: 1
   :widths: 14 24 16 46

   * - Key
     - Type
     - When omitted
     - Meaning
   * - ``path``
     - list of str
     - required
     - The field's path of keys, such as ``["inputs", "tokens"]``.
   * - ``mode``
     - ``"replicate"`` or ``"slice"``
     - ``"replicate"``
     - ``"replicate"``: every member receives the whole field.
       ``"slice"``: member ``i`` of ``n`` receives elements
       ``size * i // n`` up to ``size * (i + 1) // n`` along ``axis``; a part
       may be empty.
   * - ``axis``
     - int or None
     - None
     - The dimension ``"slice"`` divides: required for it, nonnegative, and
       None for the other modes.

An output record:

.. list-table::
   :header-rows: 1
   :widths: 14 24 16 46

   * - Key
     - Type
     - When omitted
     - Meaning
   * - ``mode``
     - ``"leader"``, ``"sum"`` or ``"concat"``
     - ``"leader"``
     - ``"leader"``: the partition's leader supplies the output. ``"sum"``:
       the members' outputs are summed. ``"concat"``: each member supplies its
       own part, joined along ``axis`` in member order.
   * - ``axis``
     - int or None
     - None
     - The dimension ``"concat"`` joins along: required for it, nonnegative,
       and None for the other modes.

Example
~~~~~~~

Two replicas of four ranks. Every rank computes with whole parameters and
trains on its own part of each batch. Within a replica, each rank stores a
quarter of every parameter's weights, gradients and optimizer state, and rank
``r + 4`` stores the same quarter as rank ``r``; a parameter whose weight
storage is under 2 MiB stays whole on every rank.

.. code-block:: python
   :name: parallelism-declaration-example

   parallelism = {
       "ranks_per_replica": 4,
       "num_replicas": 2,
       "groups": {"replica": [[0, 1, 2, 3]]},
       "default_parameter_placement": {"kind": "Replicated"},
       "parameter_placements": {},
       "default_parameter_storage": {
           "group": "replica",
           "weights": "sharded",
           "gradients": "sharded",
           "optimizer": "sharded",
           "min_shard_bytes": 2097152,
       },
       "parameter_storage": {},
       "inputs": {
           "version": 1,
           "partitions": {
               "name": "inputs",
               "groups": [[0], [1], [2], [3], [4], [5], [6], [7]],
           },
           "fields": [],
           "loss": "leader",
           "outputs": {},
       },
   }
