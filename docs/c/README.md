# C API guide

The simulator, the planner and the runtime ship as one library,
`libshadowspill`. They expose narrow C ABIs with caller-owned input and output
buffers; the public headers are authoritative for exact layouts, constants and
signatures.

The C pages remain flat because each one maps to a boundary a caller uses
independently, whether or not it is a separate library; the table below is both
the inventory and the reference index.

| Component | Header | Reference | What the page covers |
|---|---|---|---|
| Runtime | [`runtime.h`](../../csrc/include/shadowspill/runtime.h) | [Runtime API](runtime.md) | Creating a runtime and its pools and routes, reserving the records a hot path must never allocate, admitting a plan, the two task boundaries, allocation scopes, tracing, and the statistics and first-failure snapshots |
| Admission replay | [`admission_replay.h`](../../csrc/include/shadowspill/admission_replay.h) | [Runtime API](runtime.md#admission-replay) | Replaying a schedule's pool ownership transitions through the production allocator policy, with no backend and no device |
| Backend | [`backend.h`](../../csrc/include/shadowspill/backend.h) | [Backends](backends.md) | The driver-level call table a provider shared object implements, the two symbols it exports, and what a new provider directory has to build |
| Planner | [`planner.h`](../../csrc/include/shadowspill/planner.h) | [Planner API](planner.md) | The planning question in indexed form and the certification a schedule passes whichever search found it: exact schedule admission, the admission operations a schedule implies, the lease lifetimes those resolve to, and fixed-offset placement |
| PressureFit | [`pressurefit/pressurefit.h`](../../csrc/include/shadowspill/pressurefit/pressurefit.h) | [PressureFit API](pressurefit.md) | The search that ships: its options and the three policy axes, its result and per-candidate diagnostics, its preflight, and the shared placed-plan record |
| Simulator | [`simulator.h`](../../csrc/include/shadowspill/simulator.h) | [Simulator API](simulator.md) | The one call that times an already selected schedule, and the intervals, peaks, stalls and capacity shortfalls it reports |
| PyTorch adapter | [`pytorch_adapter.h`](../../csrc/adapter/pytorch/include/shadowspill/pytorch_adapter.h) | [PyTorch adapter](pytorch-adapter.md) | Only what needs PyTorch: bootstrap and the physical ledger, the pluggable-allocator callbacks, storage validation, and the stream-wrapping task boundaries |

## ABI use

Everything in `libshadowspill` is built from one tree and versions as one
thing: `SHADOWSPILL_ABI_VERSION`, in `<shadowspill/shadowspill.h>`. Compare it
with `shadowspill_abi_version()` at load time, before passing any struct.

```c
#include <shadowspill/shadowspill.h>

if (shadowspill_abi_version() != SHADOWSPILL_ABI_VERSION) {
    /* Reject the loaded library before passing any structs. */
}
```

What versions separately is what is compiled separately: the backend
contract in `<shadowspill/backend.h>`, including the profiler a backend supplies,
and the PyTorch adapter. Check those against the plugin you loaded. Do not
hardcode a numeric ABI value.

Every function in `libshadowspill` returns a `ShadowSpillStatus` unless its
signature is explicitly `void` or it is an accessor, constructor, or query
whose return value is the answer itself - `shadowspill_abi_version`,
`shadowspill_status_string`, `shadowspill_failure_reason_string`,
`shadowspill_backend_is_valid`, `shadowspill_task_id`,
`shadowspill_task_trace_label`, `shadowspill_pressurefit_best_placed_create`, and
`shadowspill_planner_struct_size`. The two separately compiled contracts
differ: a backend's table entries and its two exported symbols return `int`,
and the adapter adds two more calls whose return is the answer,
`shadowspill_pytorch_backend_malloc` (the address) and
`shadowspill_pytorch_profile_range_begin` (the range).
One vocabulary covers the whole library: the three codes
every component agrees on sit at 0-2, and each component owns a band after
that, so a status decodes to exactly one meaning without knowing which
component produced it. `shadowspill_status_string()` maps any of them to a
stable human-readable category; retain structured result fields for
diagnostics.

## Reading a header

`runtime.h` is the large one, and it is ordered rather than alphabetical.
Section banners divide it into vocabulary, the descriptions a caller fills in,
the diagnostic records the runtime fills in, and then the calls themselves in
the order a program uses them: lifecycle, pools, objects, admitting a plan,
task boundaries, allocation scopes, telemetry, and finally waiting and
inspection. Grep for a banner rather than a name if you do not know what you
are looking for yet.

`pytorch_adapter.h` follows the same convention at a smaller scale, in the
order its reference page describes.

`planner.h` carries no banners. It is ordered too, but in a repeating shape
rather than in phases: each problem's structures come first and the call that
consumes them immediately after, from schedule admission through admission
operations and lease lifetimes to placement.

`pressurefit/pressurefit.h` sits beside it and holds one search's own surface,
in the order a caller meets it: the three policy axes, the options naming them,
the preflight, the result and its per-candidate diagnostics, then the calls and
the shared placed-plan record. A search implemented elsewhere would ship a
header of the same shape and reuse `planner.h` unchanged, which is what keeps
the two files apart. The remaining headers are small enough to read start to
finish.

## Ownership rules

- Input arrays are borrowed for the documented call or admitted-plan lifetime.
- Result buffers are caller-owned unless the header says the result allocates
  them and names a matching destroy function.
- Runtime handles own their internal records, events, streams, worker, and
  pool arenas until close/destroy.
- Backend `state` pointers are borrowed and must outlive the runtime.
- Distinct simulator results and admission-replay workspaces may be used by
  different threads; one workspace is not shared concurrently.

## Platforms

The public headers are POSIX-free: no pthreads, no `<stdatomic.h>`, no
platform types. A consumer needs `<stdint.h>`, and `<stddef.h>` for the
simulator, backend and adapter headers, and nothing else; the export
declaration follows the platform - `__declspec(dllimport)` on Windows unless
the translation unit is building the library itself, which the build says with
`SHADOWSPILL_BUILDING`.

The implementation is a different matter. It uses pthreads and C11 atomics
directly, and reaches the operating system for exactly three things - a
monotonic clock, a thread yield, and a thread name - which
`csrc/src/common/platform.h` supplies for both platforms. So a Windows build
needs a toolchain that provides pthreads and `<stdatomic.h>`: MinGW-w64 or
clang does, MSVC needs `/experimental:c11atomics` and a pthreads shim. Linux
is what CI builds and what every gate here runs on; Windows is portable by
construction rather than by test.

## Build boundaries

`libshadowspill` and the mock backend build without PyTorch or a
device-provider SDK. Provider code is confined to `csrc/backends/` and the
PyTorch adapter, which stay separate libraries for that reason. See the
[C tree guide](../../csrc/README.md) for source layout and build
dependencies.
