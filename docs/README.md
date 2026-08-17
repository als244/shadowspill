# ShadowSpill documentation

This directory describes the current ShadowSpill implementation. The
documentation is split by contract so readers do not have to infer whether a
statement applies to Python, C, or the framework-neutral design.

## Start here

- [Architecture overview](architecture/overview.md) — component ownership,
  data flow, and supported scope.
- [Python guide](python/README.md) — installation, planning, execution,
  artifacts, diagnostics, and public APIs.
- [C guide](c/README.md) — compiled runtime, planner, simulator, PyTorch
  adapter, and backend contracts.
- [Development guide](development/README.md) — repository structure, testing,
  naming, and documentation rules.

## Architecture

- [Intermediate representation](architecture/ir.md)
- [PyTorch capture and lowering](architecture/lowering.md)
- [Memory runtime](architecture/memory-runtime.md)
- [Planning and physical admission](architecture/planning.md)
- [Simulation](architecture/simulation.md)

## Historical evidence

[Engineering investigations](investigations/README.md) preserve root-cause
evidence for bugs and performance work. They describe the revision under
investigation and are not normative specifications. When an investigation and
an architecture or API page differ, the architecture or API page is the
current contract.

## Documentation policy

Public behavior is documented here and tested against exported Python names,
public C headers, and local links. Source files and installed headers remain
authoritative for exact signatures, ABI constants, and cache schema labels so
the public prose has no duplicate authority.
