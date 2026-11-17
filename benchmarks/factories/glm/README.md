# Shared GLM recipes

Model definitions and checkpoint tensor mappings live in
[the model families](../../../packages/workloads/src/shadowspill_workloads/models). This package owns their caller-side policy.

- `quickstart.py`: GLM-5.3 and Flash factories for the ordinary phase runner.
- `factory.py`, `model.py`: meta model construction, precision and build options.
- [`initialization/glm.py`](../../../packages/workloads/src/shadowspill_workloads/initialization/glm.py): bounded checkpoint loading or reproducible random values.
- `inputs.py`: synthetic inputs across candidate batch sizes.
- `data.py`: real-text batches, objectives and the fixed-batch learning check.
- [GLM partition policy](../../../packages/workloads/src/shadowspill_workloads/policies/glm.py): optional per-layer partitioning.

Use the public [model catalog](../../../docs/python/catalog.md) for
training through the service. These factories serve direct phase benchmarks.
