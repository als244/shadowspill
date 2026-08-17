# Python guide

The Python package provides model-state relocation, PyTorch capture and
lowering, reusable planning artifacts, PressureFit, diagnostics, and callable
execution.

## Guides

- [Quickstart](quickstart.md)
- [Planning cache](planning-cache.md)
- [PyTorch allocator integration](allocator.md)

## API reference

- [Frontend and lifecycle](api/frontend.md)
- [Reusable planning artifacts](api/artifacts.md)
- [Planning and step diagnostics](api/diagnostics.md)
- [Framework-neutral Python APIs](api/core.md)

The supported user entrypoints are imported from `shadowspill.memory` and
`shadowspill.pytorch`. The lower-level modules `shadowspill.ir`,
`shadowspill.planner`, `shadowspill.simulator`, and `shadowspill.runtime` are
public for tooling, experiments, and independent planning.
