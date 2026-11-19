# ShadowSpill engine

Build, profile, plan and execute model computations within an explicit memory
budget. The Python package is `shadowspill`; its public phase APIs and native
planner/runtime are installed together.

See the [repository guide](../../docs/development/repository.md) for setup and
[phase APIs](../../docs/python/phases.md) for use. Built-in model definitions live
in the separate [workloads package](../workloads/README.md).

For a local planner build without accelerator discovery or the PyTorch adapter:

```bash
pip install ./packages/engine --config-setting cmake.define.SHADOWSPILL_CPU_ONLY=ON
```

This installs the native CPU planner, simulator and mock runtime. It
supports saved-artifact inspection and replanning; preparing models and executing
their device kernels requires the qualified server build. The default source
build retains automatic backend discovery. Explicit provider selections conflict
with `SHADOWSPILL_CPU_ONLY=ON` and are rejected.
