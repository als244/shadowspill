# ShadowSpill documentation

Start with [installation](development/repository.md#setup), then choose a path.

| I want to… | Start here |
| --- | --- |
| Train through the public service | [Complete client example](examples/client-training.md), [client API](python/client.md): settings, global batches, evaluation and resume |
| Run a supplied text application | [Llama](../examples/training/llama3/README.md), [Qwen3](../examples/training/qwen3.md), [Qwen3.5 LoRA](../examples/training/qwen35_lora.md) |
| Use a custom model or call phases directly | [Custom models](python/custom-models.md), [phase guide](python/phases.md), [complete example](examples/generic-training.md) |
| Tune, diagnose or extend the system | [Advanced guides](python/advanced/README.md): chunking, retention, artifacts, traces and operators |

`ServiceClient` creates or reconnects to a run. `TrainingClient` submits runtime
operations to ready workers. The service uses the existing build/profile/plan/
execute engine, and the ordinary client requires no model or GPU framework.
[Definition hooks](python/definitions.md) support installed custom models and
objectives through the same interface as supplied models.

Migration and qualification are unfinished; see [current coverage](python/client.md#current-coverage).
The [recipe runner](../packages/workloads/src/shadowspill_workloads/recipes/README.md)
and saved-Plan `Trainer` still serve unmigrated callers. Their guides remain
available until those callers move to the public service.

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
