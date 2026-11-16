# ShadowSpill

Train PyTorch models within a memory budget. ShadowSpill plans which tensors to
keep on the GPU, spill to other storage, and recompute during backward.

The training service runs on your GPU machine. Your client sends batches and
receives results, with only Python and NumPy installed locally.

- [Start training](docs/examples/client-training.md) — a complete text-training example.
- [Client API](docs/python/client.md) — setup, updates and checkpoints.
- [Advanced usage](docs/python/advanced/README.md) — custom models, preparation,
  split updates, diagnostics and direct engine APIs.
- [Installation](docs/development/repository.md#setup) · [All documentation](docs/README.md)
