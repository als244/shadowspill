# Python guide

The Python package provides model-state import, PyTorch capture and
lowering, reusable planning artifacts, PressureFit, diagnostics, and callable
execution.

The directory is intentionally split between task-oriented guides and the
`api/` reference.

## Guides

- [Quickstart](quickstart.md)
- [Planning cache](planning-cache.md)
- [PyTorch allocator integration](allocator.md)
- [Errors, failures, and cleanup](failures.md)
- [Interpreting a PlanReport](plan-report.md)
- [Interpreting StepResult diagnostics](step-diagnostics.md)
- [Program and annotated-plan JSON](planning-json.md)
- [Practical examples](../examples/README.md)

## API reference

- [Frontend and lifecycle](api/frontend.md)
- [Reusable planning artifacts](api/artifacts.md)
- [Planning and step diagnostics](api/diagnostics.md)
- [Framework-neutral Python APIs](api/neutral.md)

The supported user entrypoints are imported from `shadowspill.memory` and
`shadowspill.pytorch`. The lower-level modules `shadowspill.ir`,
`shadowspill.planner`, `shadowspill.simulator`, and `shadowspill.runtime` are
public for tooling, experiments, and independent planning.
