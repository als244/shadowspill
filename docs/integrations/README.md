# Optional operator-library adapters

The core compiler accepts generic operator descriptions and build policies.
An optional adapter translates a library's own metadata into those controls;
it does not make the engine depend on a model architecture.

- [MLOps](mlops.md): independent axes, additive gradients, layout permissions
  and prepared external kernels.

To integrate another library, start with [operator integration](../python/advanced/operator-integration.md)
and [operator preparation](../python/advanced/operator-preparation.md). Provider semantics
and kernel ownership stay with the library; the consuming adapter and its
compiler integration tests belong in this repository.
