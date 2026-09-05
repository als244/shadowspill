# The C tree

`csrc/` builds one library, `libshadowspill`, plus the pieces that are
genuinely pluggable: the device backends and the PyTorch adapter.

```text
csrc/
├── include/shadowspill/   every public header the library exports
├── src/
│   ├── common/            what all three share: the status decoder and the
│   │                      three calls POSIX and Windows spell differently
│   ├── simulator/         deterministic schedule evaluator
│   ├── planner/           PressureFit candidate selection over the simulator;
│   │   └── admission/     derives the pool operations a schedule implies,
│   │                      replays them, resolves lease lifetimes, and places
│   │                      the leases at fixed addresses
│   └── runtime/           pools, leases, objects, transfers, and the worker,
│       ├── memory/          split by subsystem: ranges, pools, leases,
│       ├── objects/          retirement
│       ├── tasks/
│       ├── transfers/
│       ├── sync/
│       ├── plan/
│       └── telemetry/
├── backends/              dlopened device backends: mock and provider
└── adapter/pytorch/       narrow allocator/storage bridge into PyTorch
    └── lifecycle/         bootstrap, close, and the physical-memory ledger
```

Everything under `src/` compiles into one shared object. The simulator, the
planner and the runtime were three, with a strict dependency order between them
and nothing that ever linked them apart; separating them bought no independent
deployment and cost an ABI and a status vocabulary per component. They now
share both: one `SHADOWSPILL_ABI_VERSION`, one `ShadowSpillStatus`.

Backends stay separate because that is what they are for — each is dlopened and
compiled against the backend contract alone, and a provider backend needs a
toolchain the rest of the tree must not require. The PyTorch adapter stays
separate because it links libtorch, which planning-only callers must not be
made to carry. All three keep their own ABI versions, since they are genuinely
compiled elsewhere.

`src/common/platform.h` holds the three things the library asks of the
operating system that POSIX and Windows spell differently: a monotonic clock,
a thread yield, and a thread name. Everything else it needs - threads,
mutexes, atomics - comes from pthreads and `<stdatomic.h>`, which a Windows
build gets from its toolchain rather than from a shim here.

Public headers live in `include/shadowspill/`. A private header named
`internal.h` belongs to the directory holding it, and is included by path from
anywhere else, so `"internal.h"` always means this directory's.

The [C API guide](../docs/c/README.md) documents ownership, threading, and each
public component boundary.
