# Simulator C API

The public interface is declared by
`simulator/include/shadowspill/simulator.h` and implemented by
`libshadowspill_simulator.so`.

## ABI and ownership

- Call `shadowspill_simulator_abi_version()` before supplying records. The
  current ABI is version 1.
- `ShadowSpillSimulationProgram` borrows every pointed-to array only for the
  duration of `shadowspill_simulate`.
- The caller owns task-interval, transfer-interval, and device-peak buffers in
  `ShadowSpillSimulationResult`. Their capacities must cover the corresponding
  program counts.
- The simulator allocates only private scratch memory, retains no caller
  pointer, performs no I/O, and has no global mutable state.
- Concurrent calls are safe when their result buffers are distinct.

Every count, offset array, dense identity, enum, and capacity is validated
before simulation. A nonzero status leaves ownership unchanged and populates
the result's structured error fields. Use
`shadowspill_simulation_status_string()` only for human-facing text.

## Synchronization

This library is a deterministic offline simulator. It creates no threads and
uses no accelerator or framework synchronization primitive. Resource lanes and
transfer lanes are logical schedule resources measured in integer nanoseconds.
