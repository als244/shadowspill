# Python guide

The public workflow separates build, measurement, CPU planning and fresh-worker
execution. Each phase produces an artifact that can be saved and reused. Trainer
runs an already prepared Plan through Executor. This directory contains guides
and the `api/` reference.

## Guides

How to use the system and how to read what it produces.

- [Training with saved plans](../examples/generic-training.md)
- [Independent build, profile, plan and execution APIs](../architecture/offline-pipeline.md)
- [Artifact store](artifact-store.md)
- [PyTorch allocator integration](allocator.md)
- [Errors, failures, and cleanup](failures.md)
- [Interpreting a PlanReport](plan-report.md)
- [PlanReport field reference](plan-report-fields.md)
- [PressureFit cut traces](pressurefit-trace.md)
- [Interpreting StepResult diagnostics](step-diagnostics.md)
- [program and annotated-plan JSON](planning-json.md)
- [Figures over a step search](plots.md)
- [Pool occupancy over a step](occupancy.md)
- [Practical examples](../examples/README.md)

## API reference

Every public symbol, its arguments and its result, one page per layer,
ordered outward from the framework-neutral layer to the PyTorch frontend.

- [Framework-neutral Python APIs](api/neutral.md)
- [Reusable planning artifacts](api/artifacts.md)
- [Trainer and Executor](api/training.md)
- [Distributed planning and training](api/distributed.md)
- [Frontend and lifecycle](api/frontend.md)
- [SSD spill storage](api/ssd.md)
- [Planning and step diagnostics](api/diagnostics.md)
- [Timing: the step on the device clock](api/timing.md)

The phase entrypoints are imported from `shadowspill`: builders, measurement,
profile application, planning, artifact classes, Executor and Trainer. The older
entrypoints in `shadowspill.memory` and `shadowspill.pytorch` remain during
repository caller migration. `shadowspill.training` supplies training-loop tools.
`shadowspill.ssd` and `shadowspill.network` configure additional spill storage.
The lower-level packages `shadowspill.ir`,
`shadowspill.step`, `shadowspill.task`, `shadowspill.store`,
`shadowspill.planner`, `shadowspill.search`,
`shadowspill.simulator`, `shadowspill.diagnostics` and `shadowspill.runtime`
are public for tooling, experiments, and independent planning. None of them
imports a framework, so any of them can be read without one.
`shadowspill.profiling` is the GPU measurement phase and imports PyTorch; artifact
loading, profile application, planning and reporting remain CPU-only.
