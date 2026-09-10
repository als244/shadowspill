# The C tree

`csrc/` builds one library, `libshadowspill`, plus the pieces that are
genuinely pluggable: the device backends and the PyTorch adapter.

```text
csrc/
├── include/shadowspill/   every public header the library exports
│   └── pressurefit/       the shipped search's own header, beside the generic
│                          planner header rather than inside it
├── src/
│   ├── common/            what all three share: the status decoder and the
│   │                      calls POSIX and Windows spell differently
│   ├── simulator/         deterministic schedule evaluator
│   ├── planner/           the planning question, and answering it
│   │   ├── admission/     search-agnostic certification: derives the pool
│   │   │                  operations a schedule implies, replays them,
│   │   │                  resolves lease lifetimes, and places the leases at
│   │   │                  fixed addresses
│   │   └── search/
│   │       ├── toolkit/   what any search may call: schedule binding and
│   │       │              digests
│   │       └── algorithms/
│   │           └── pressurefit/  the search that ships, over the simulator
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
    ├── include/shadowspill/  its one public header
    ├── lifecycle/         bootstrap, close, and the physical-memory ledger
    ├── allocator/         the callbacks PyTorch's pluggable allocator makes
    ├── failure/           what a failed call latches, and the report it makes
    ├── tasks/             the task boundary: ranges, scopes, the action batch
    └── storage/           PyTorch storages over runtime leases
```

Everything under `src/` compiles into one shared object. The simulator, the
planner and the runtime have a strict dependency order between them and nothing
that links them apart, so they share one `SHADOWSPILL_ABI_VERSION` and one
`ShadowSpillStatus` rather than an ABI and a status vocabulary each.

Backends are separate because that is what they are for: each is dlopened and
compiled against the backend contract alone, and a provider backend needs a
toolchain the rest of the tree must not require. The PyTorch adapter is
separate because it links libtorch, which planning-only callers must not be
made to carry. Both keep their own ABI version, being genuinely compiled
elsewhere.

`src/common/platform.h` holds what the library asks of the operating system
that POSIX and Windows spell differently: a monotonic clock, a thread yield, a
thread name, and the logical CPU count. Everything else it needs - threads,
mutexes, atomics - comes from pthreads and `<stdatomic.h>`, which a Windows
build gets from its toolchain rather than from a shim here.

Public headers live in `include/shadowspill/`. A private header named
`internal.h` belongs to the directory holding it, and is included by path from
anywhere else, so `"internal.h"` always means this directory's.
The adapter adds one refinement: a directory whose header the C++ storage
operators include -- `allocator/`, `failure/`, `tasks/`, `storage/` -- keeps
that header free of C11 atomics, and its C files include the adapter's state
header, `../internal.h`, themselves.

The [C API guide](../docs/c/README.md) documents ownership, threading, and each
public component boundary.
