# C API guide

ShadowSpill's compiled components expose narrow C ABIs with caller-owned input
and output buffers. Public headers are authoritative for exact layouts,
constants, and signatures.

| Component | Header | Reference |
|---|---|---|
| Runtime | [`runtime.h`](../../csrc/runtime/include/shadowspill/runtime.h) | [Runtime API](runtime.md) |
| Admission replay | [`admission_replay.h`](../../csrc/runtime/include/shadowspill/admission_replay.h) | [Runtime API](runtime.md#admission-replay) |
| Backend | [`backend.h`](../../csrc/runtime/include/shadowspill/backend.h) | [Backends](backends.md) |
| Profiler | [`profiler.h`](../../csrc/runtime/include/shadowspill/profiler.h) | [Backends](backends.md#profiler) |
| Planner | [`planner.h`](../../csrc/planner/include/shadowspill/planner.h) | [Planner API](planner.md) |
| Simulator | [`simulator.h`](../../csrc/simulator/include/shadowspill/simulator.h) | [Simulator API](simulator.md) |
| PyTorch adapter | [`pytorch_adapter.h`](../../csrc/pytorch_adapter/include/shadowspill/pytorch_adapter.h) | [PyTorch adapter](pytorch-adapter.md) |

## ABI use

Compile against the ABI macro from the installed header and compare it with
the component's `*_abi_version()` function at load time. Do not hardcode a
numeric ABI value outside the component that owns it.

```c
#include <shadowspill/simulator.h>

if (shadowspill_simulator_abi_version() !=
    SHADOWSPILL_SIMULATOR_ABI_VERSION) {
    /* Reject the loaded component before passing any structs. */
}
```

Every public function returns a component status unless its signature is
explicitly `void`. Use the corresponding `*_status_string()` function for a
stable human-readable category and retain structured result fields for
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

The simulator, planner, neutral runtime, and mock backend build without
PyTorch or CUDA. Provider code is confined to runtime backend directories and
the PyTorch adapter. See the [compiled component guide](../../csrc/README.md)
for source layout and build dependencies.
