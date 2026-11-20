# ShadowSpill workloads

Optional model architectures, checkpoint mappings, input processors and registered
training components, including the MLOps AdamW optimizer. Import them as
`shadowspill_workloads`. The engine can prepare and run custom models without
installing this catalog.

The catalog uses the MLOps operation library, which this package owns:
`scripts/setup_mlops.sh --python PATH` installs it from a checkout named `mlops`
beside this repository when there is one, warning when that checkout is not at
the pinned revision, and otherwise at the pinned revision.

See the [built-in components](../../docs/python/catalog.md),
[client guide](../../docs/python/client.md), and
[repository setup](../../docs/development/repository.md).
For contributors, [workload organization](../../docs/development/workloads.md)
explains the package hierarchy and the steps for adding a built-in family.
