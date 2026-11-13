# Training a custom model

For a model already covered by the [training runner](../../workloads/recipes/README.md),
use `train(model, config)`. For a different PyTorch model, start from the
[language-model phase example](../examples/generic-training.md) and replace its
model, objective, inputs and initialization. The same compiler, profiler, planner
and Trainer handle both paths; the engine does not require a catalog entry.

## Supply the training computation

Construct your `torch.nn.Module` and optimizer, preferably on `meta` when full
weights would consume too much memory. Pass a representative call and an
objective to `build_step`:

```python
from shadowspill import build_step

def objective(model, inputs, targets, *, normalizer):
    return model.loss(inputs, targets, reduction="sum") * normalizer

program = build_step(
    model, optimizer=optimizer, objective=objective,
    example_args=(example_inputs, example_targets),
    example_kwargs={"normalizer": 1 / tokens_per_update},
    num_examples=microbatches_per_update,
)
```

The names above are your model/data values. The objective must return a scalar.
Trainer accumulates its gradients without inserting another average: normalize
over the complete optimizer update, including all ranks when distributed.
For classification or regression, use the corresponding sample count instead
of a token count. Parameters belong to the optimizer; persistent non-parameter
tensors belong in registered buffers.

The Program specializes shapes, dtypes and Python metadata from the example.
Use `example_calls` for heterogeneous calls, or build distinct Programs for
different microbatch shapes. Measure each implementation and compare admitted
Plans with [candidate sweeps](api/planning.md). Checkpoint partitioning and
execution chunking are optional [build policies](api/frontend.md), separate
from your model's forward implementation.

## Supply data and initial state

An input source yields one mapping per optimizer update:

```python
{
    "args": [(inputs_0, targets_0), (inputs_1, targets_1)],
    "kwargs": [{"normalizer": scale}, {"normalizer": scale}],
}
```

Each entry is one complete call matching the built Program. CPU tensors are
accepted; the runtime places them. A resumable source implements `state_dict()`
and `load_state_dict(state)` so checkpoints can restore its cursor.

After planning, construct `Trainer` in a fresh execution worker. Its
`initialize(fill_tensor)` callback receives a `TensorChunk` request and a CPU
destination view. Fill the requested slice of the named tensor's physical storage;
the request includes shape/stride metadata for noncontiguous tensors. Copy from
a checkpoint or generate directly, without materializing the full model. The
offset is global even for a distributed shard. The Program supplies optimizer-state
initializers. See the
[Trainer initialization reference](api/training.md#start-or-resume).

Then call `trainer.fit(source, steps=..., run_dir=...)`. Build a separate
`build_forward` Program for evaluation and admit it alongside training before
passing its Plan as `evaluation_plan`. See [distributed setup](api/distributed.md)
for parameter ownership, rank-local inputs and communication calibration.

## Check the integration

Start with a small model and short inputs. Compare a complete update with eager
training, including gradients and optimizer state; test checkpoint reload in a
fresh worker. Add chunking and recomputation policies after that baseline works,
then measure the actual workload under its intended memory budget.

Ordinary capturable PyTorch operations need no provider integration. Custom
operations need capture/derivative support; splitting them also requires explicit
axis and reduction declarations. Follow [operator integration](operator-integration.md)
and, for saved external kernels, [operator preparation](operator-preparation.md).

To contribute another supported recipe to the convenience API, implement the
small [adapter hooks](../../workloads/recipes/configuration.md#adapter-boundary)
and expose its model source in the workload package. Keep model knowledge there;
the shared runner owns subprocesses, artifacts, search and execution.
