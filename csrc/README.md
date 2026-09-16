# The C tree

`csrc/` builds one library, `libshadowspill`, plus the pieces that are
genuinely pluggable: the device backends and the PyTorch adapter.

```text
csrc/
├── include/shadowspill/   every public header the library exports
│   ├── runtime/           the runtime API by subsystem, included by the
│   │                      umbrella `runtime.h`: vocabulary, descriptions,
│   │                      diagnostics, lifecycle, pools, objects, plan,
│   │                      tasks, telemetry, timing
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
│   │               ├── candidates/  one candidate's stages, and the layers
│   │               │                beneath them: workspaces, memos, the
│   │               │                plan, the repairs, the work accounting
│   │               ├── problem/     one problem prepared once: its buffers,
│   │               │                the facts derived from the program, the
│   │               │                floor it may not go below, the placement
│   │               ├── residency/   reducing what is resident, one cut at a
│   │               │                time: spans, cuts, the index, the tree,
│   │               │                the heap, the loop that drives them
│   │               └── schedule/    where the transfers go: pressure,
│   │                                triggers, the clamp, and what is emitted
│   └── runtime/           pools, leases, objects, transfers, and the worker,
│       │                  split by subsystem; the runtime object itself is
│       │                  opened, closed, grown and read in four files here
│       ├── memory/          the arena and what it hands out
│       │   ├── memory_pool/   arena, records, locks, leases, causal handoff
│       │   └── allocations/   one allocation: indexed, owned, made, freed
│       ├── objects/          the table, its owners, its allocations, and
│       │                     handing an object to the caller
│       ├── tasks/            the table, the record, admission, the handles,
│       │                     the boundaries, and the scopes between them
│       ├── transfers/        the lanes, and what is in flight on each
│       ├── sync/             event leases and their pools, completion tracking,
│       │                     the quiescence wake-up, and the markers a caller
│       │                     times its own work with
│       ├── plan/             a plan's admission, its lifetime, its residue
│       ├── telemetry/        the trace rings, the profiler, the statistics
│       └── worker/           one action handled, dispatched, completed
├── backends/              dlopened device backends: mock and provider
└── adapter/pytorch/       narrow allocator/storage bridge into PyTorch
    ├── include/shadowspill/  its one public header
    ├── lifecycle/         bootstrap, close, and the physical-memory ledger
    ├── allocator/         the callbacks PyTorch's pluggable allocator makes
    ├── failure/           what a failed call latches, and the report it makes
    ├── tasks/             the task boundary a planned task runs between,
    │                      the thread's task range, and the scopes outside one
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

Public headers live in `include/shadowspill/`. `runtime.h` is an umbrella: it
includes one header per subsystem from `include/shadowspill/runtime/`, so a
caller may take the whole API as before or just the part it uses, and each part
compiles on its own. A private header named `internal.h` belongs to the
directory holding it, and is included by path from anywhere else, so
`"internal.h"` always means this directory's.
The adapter adds one refinement: a directory whose header the C++ storage
operators include -- `allocator/`, `failure/`, `tasks/`, `storage/` -- keeps
that header free of C11 atomics, and its C files include the adapter's state
header, `../internal.h`, themselves.

The [C API guide](../docs/c/README.md) documents ownership, threading, and each
public component boundary.
