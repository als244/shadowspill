# C API guide

The simulator, the planner and the runtime ship as one library,
`libshadowspill`. They expose narrow C ABIs with caller-owned input and output
buffers; the public headers are authoritative for exact layouts, constants and
signatures.

The C pages remain flat because each one maps to a boundary a caller uses
independently, whether or not it is a separate library; the table below is both
the inventory and the reference index.

| Component | Header | Reference |
|---|---|---|
| Runtime | [`runtime.h`](../../csrc/include/shadowspill/runtime.h) | [Runtime API](runtime.md) |
| Admission replay | [`admission_replay.h`](../../csrc/include/shadowspill/admission_replay.h) | [Runtime API](runtime.md#admission-replay) |
| Backend | [`backend.h`](../../csrc/include/shadowspill/backend.h) | [Backends](backends.md) |
| Profiler | [`profiler.h`](../../csrc/include/shadowspill/profiler.h) | [Backends](backends.md#profiler) |
| Planner | [`planner.h`](../../csrc/include/shadowspill/planner.h) | [Planner API](planner.md) |
| Simulator | [`simulator.h`](../../csrc/include/shadowspill/simulator.h) | [Simulator API](simulator.md) |
| PyTorch adapter | [`pytorch_adapter.h`](../../csrc/adapter/pytorch/include/shadowspill/pytorch_adapter.h) | [PyTorch adapter](pytorch-adapter.md) |

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

What versions separately is what is compiled separately: the three backend
contracts in `<shadowspill/backend.h>`, the profiler struct a backend supplies,
and the PyTorch adapter. Check those against the plugin you loaded. Do not
hardcode a numeric ABI value.

Every public function returns a `ShadowSpillStatus` unless its signature is
explicitly `void`. One vocabulary covers the whole library: the three codes
every component agrees on sit at 0-2, and each component owns a band after
that, so a status decodes to exactly one meaning without knowing which
component produced it. `shadowspill_status_string()` maps any of them to a
stable human-readable category; retain structured result fields for
diagnostics.

## Ownership rules

- Input arrays are borrowed for the documented call or admitted-plan lifetime.
- Result buffers are caller-owned unless the header says the result allocates
  them and names a matching destroy function.
- Runtime handles own their internal records, events, streams, worker, and
  pool arenas until close/destroy.
- Backend context pointers are borrowed and must outlive the runtime.
- Distinct simulator results and admission-replay workspaces may be used by
  different threads; one workspace is not shared concurrently.

## Build boundaries

`libshadowspill` and the mock backend build without PyTorch or a
device-provider SDK. Provider code is confined to `csrc/backends/` and the
PyTorch adapter, which stay separate libraries for that reason. See the
[C tree guide](../../csrc/README.md) for source layout and build
dependencies.
