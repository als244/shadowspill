# ShadowSpill documentation

## Start here

[Install ShadowSpill](development/repository.md#setup), then run the
[text-training example](examples/client-training.md). The service prepares the
model and workers; the client submits batches and reads results.

- [Client API](python/client.md): training settings, updates and checkpoints.
- [Models, optimizers and objectives](python/catalog.md): built-ins, LoRA and freezing.
- [Examples](examples/README.md): standalone training demos and custom workflows.
- [Advanced usage](python/advanced/README.md): preparation tuning, split updates,
  extensions, recovery and diagnostics.

## Reference

Exact APIs, the engine's design, the C interface and contributor guides.

- [Python guides and API reference](python/README.md)
- [Engine architecture](architecture/README.md)
- [C API](c/README.md)
- [Development](development/README.md)
- [Optional integrations](integrations/README.md)
