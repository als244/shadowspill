# Python guide

The Python package covers model-state import, PyTorch capture and lowering,
reusable planning artifacts, planning, diagnostics, and callable execution. The
directory is split between task-oriented guides and the `api/` reference.

## Guides

How to use the system and how to read what it produces.

- [Quickstart](quickstart.md)
- [Artifact store](artifact-store.md)
- [PyTorch allocator integration](allocator.md)
- [Errors, failures, and cleanup](failures.md)
- [Interpreting a PlanReport](plan-report.md)
- [PlanReport field reference](plan-report-fields.md)
- [Interpreting StepResult diagnostics](step-diagnostics.md)
- [program and annotated-plan JSON](planning-json.md)
- [Figures over a step search](plots.md)
- [Practical examples](../examples/README.md)

## API reference

Every public symbol, its arguments and its result, one page per layer and
ordered outward from the framework-free core.

- [Framework-neutral Python APIs](api/neutral.md)
- [Reusable planning artifacts](api/artifacts.md)
- [Frontend and lifecycle](api/frontend.md)
- [Planning and step diagnostics](api/diagnostics.md)
- [Timing: the step on the device clock](api/timing.md)

The supported user entrypoints are imported from `shadowspill.memory` and
`shadowspill.pytorch`. The lower-level packages `shadowspill.ir`,
`shadowspill.step`, `shadowspill.store`, `shadowspill.planner`,
`shadowspill.simulator` and `shadowspill.runtime` are public for tooling,
experiments, and independent planning.
