# ShadowSpill documentation

Start with [installation](development/repository.md#setup), then choose a path.

| I want to… | Start here |
| --- | --- |
| Train a supported model end to end | [Python training](../packages/workloads/src/shadowspill_workloads/recipes/README.md): model, data, budgets, search and resume |
| Use a custom model or call phases directly | [Custom models](python/custom-models.md), [phase guide](python/phases.md), [complete example](examples/generic-training.md) |
| Tune, diagnose or extend the system | [Advanced guides](python/advanced/README.md): chunking, retention, artifacts, traces and operators |

The supported `train` runner handles the whole workflow. The phase APIs expose
build, profile, plan and execute separately; `Trainer` runs an already selected
Plan. Both paths use the same compiler, planner and runtime.

## References and development

| Directory | Contents |
| --- | --- |
| [Python API index](python/README.md) | Exact options, defaults and constraints in `python/api/` |
| [Examples](examples/README.md) | Complete workflows to adapt |
| [Architecture](architecture/README.md) | Engine design and invariants, grouped by subsystem |
| [C reference](c/README.md) | Native runtime, backends, pools, planner and simulator |
| [Development](development/README.md) | Setup, source ownership, naming and validation |
| [Optional integrations](integrations/README.md) | Adapters for specific operator libraries |

Experiment histories, qualification receipts and merge-review notes stay under
`docs/internal/`; they are separate from the public guides.
