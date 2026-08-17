# Compiled components

`csrc/` contains every compiled ShadowSpill component. Each directory owns one
API boundary and one small `CMakeLists.txt`; the repository-root build file only
orders their dependencies.

```text
csrc/
├── simulator/          deterministic schedule evaluator
├── runtime/            pools, leases, objects, transfers, worker, and backends
├── planner/            PressureFit candidate selection over the simulator
└── pytorch_adapter/    narrow allocator/storage bridge into PyTorch
```

Public C headers live under each component's `include/shadowspill/` directory.
Implementation files and private headers live under `src/`. The runtime alone
has backend implementations: `mock` supports accelerator-free testing and
`cuda` supplies device pools, streams, events, copies, and NVTX profiling.

The dependency direction is:

```text
simulator ─┐
           ├─► planner
runtime ───┘
   ▲
   └──────── PyTorch adapter
```

The simulator never calls the planner. The runtime has no PyTorch dependency.
Only `runtime/backends/cuda/` and `pytorch_adapter/` require the CUDA/PyTorch
toolchains; the simulator, planner, runtime, and mock backend build without
them.
