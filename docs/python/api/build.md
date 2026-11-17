# Build

Build captures a model and representative calls, partitions the computation and
finalizes executable tasks into a `Program`. It needs a representative GPU even
when model tensors start on CPU/meta. It does not initialize persistent training
state, measure task performance or open execution/spill pools. See the
[phase guide](../phases.md) for what follows.

## Choose a builder

| Function | Captured computation | Execute with |
| --- | --- | --- |
| `build_step` | Objective, backward and optimizer update | `Trainer` or `Executor` |
| `build_forward_backward` | Objective and backward; gradients persist, no optimizer | `Executor`, with fresh/accumulating gradient modes |
| `build_forward` | One forward call, optionally through `forward_fn` | `Executor`, or Trainer's `evaluation_plan` |

```python
from shadowspill import build_step

program = build_step(
    model, objective=objective, optimizer=optimizer,
    example_args=(features,), example_kwargs={"targets": targets},
    num_examples=4,
)
program.save("program.ssp")
```

`objective(model, *args, **kwargs)` returns a scalar loss, `(loss, metrics)`, or
[`ObjectiveResult`](frontend.md#inputs-objectives-and-partitioning). It owns normalization across
microbatches and ranks. Model and optimizer values are not changed by build;
weights and captured optimizer defaults are initialized later in the worker.
The [complete example](../../examples/generic-training.md) defines these objects.

## Training options

All arguments after `model` are keyword-only. `build_step` accepts:

<!-- api-options: packages/engine/src/shadowspill/build/step.py:build_step -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `model` | Required | PyTorch module, with CPU, meta or device tensors. |
| `objective` | Required | Scalar training objective and optional reporting metrics. |
| `optimizer` | Required | Capturable optimizer referring to this model's parameters. |
| `example_args` | `()` | Positional arguments for one complete microbatch call. |
| `example_kwargs` | `None` | Keyword arguments for that call. |
| `num_examples` | `1` | Number of rank-local calls per update with that same signature. |
| `example_calls` | `None` | Explicit sequence of `(args, kwargs)` pairs for differing call signatures. |
| `partition` | `"auto"` | Checkpoint-stage boundaries. |
| `chunking` | `None` | Optional execution regions, split axes and per-region chunk sizes. |
| `graph_pair_policy` | `None` | Default `GraphPairPolicy()`, or independent retention rules for save/recompute variants. |
| `grad_dtype` | `None` | Stored gradient dtype; default follows the parameter's logical dtype. |
| `master_dtype` | `None` | Optional separate optimizer master-weight dtype. |
| `runtime_gradient_scale` | `False` | Capture a runtime final-scale control, defaulting to 1, at each optimizer consumer. |
| `fuse_narrow_gradient_accumulation` | `True` | Fuse supported gradient production into its accumulation destination. |
| `gradient_destination_rules` | `()` | Explicit provider declarations for opaque operations that can accumulate into gradient destinations. |
| `parallelism` | `None` | Logical ranks and tensor ownership; default is local execution. |
| `resources` | `None` | Descriptions of shared backend resources required by model operators. |
| `operator_preparation` | `()` | Explicit provider adapters for saving/restoring external kernel selections. |
| `rank_map` | `None` | Worker → logical-rank mapping when build needs distributed participants. |
| `target` | `"cuda"` | Representative device for finalizing task code. |

`build_forward_backward` accepts the same options **except** `optimizer` and
`master_dtype` and `runtime_gradient_scale`. It leaves optimizer policy to the caller.

### Calls and microbatches

`num_examples` counts complete calls, not samples or tokens. With
`features.shape == (16, D)` and `num_examples=4`, one update processes 64 samples
per rank. Each runtime `args`/`kwargs` sequence must contain four entries matching
the captured signatures. The planner chooses their traversal later; it does
not change their shapes or count.

Use `example_calls=[(args0, kwargs0), (args1, kwargs1)]` **instead of**
`example_args`, `example_kwargs` and `num_examples` for heterogeneous shapes or
static metadata. Matching signatures share captures. Tensor values can vary at
execution; specialized Python values remain fixed.

### Memory and precision

`partition`, `chunking` and `graph_pair_policy` control different decisions:
checkpoint boundaries, temporary execution lifetimes, and retained intermediates
inside each save/recompute variant. None forces retained tensors to remain on
the GPU; planning selects variants and placement from the measured alternatives.
Changing these controls requires building and profiling the affected tasks.
See [frontend policies](frontend.md) and the
[partitioning example](../../examples/custom-partitioning.md).

The default graph pair uses a cost-based save policy; its recompute variant
retains checkpoint inputs and supported routing choices, regenerating other
intermediates. Explicit rules can change each variant independently.

`master_dtype=torch.float32` leaves model compute precision unchanged while
adding optimizer master weights for lower-precision parameters. Ordinary parameters
already at that precision reuse their storage; master dtype never narrows them.
`grad_dtype` controls stored gradients;
optimizer rules determine moment/state dtypes. Ordinary masters are initialized
from compute weights in bounded chunks. Trainable tensor wrappers require an
explicit master dtype and separate initialization of physical components and
`master.<parameter-name>`. See [initialization](../../architecture/state-import.md).

Custom optimizers need a tensor-only capturable update. Python tensor-value
branches, unsupported first-use state creation and step hooks are rejected;
there is no eager optimizer fallback. Runtime controls such as scheduled learning
rates must be captured as such, rather than baked-in Python constants.

## Forward-only options

`build_forward` accepts `model`, `example_args`, `example_kwargs`, `partition`,
`chunking`, `parallelism`, `resources`, `operator_preparation`, `rank_map` and
`target`, with the defaults above. It captures one call and has no objective,
optimizer, gradient dtype, call-count or graph-pair retention argument.

Its additional `forward_fn=None` option defaults to `model.forward`. A callback
`forward_fn(model, *args, **kwargs)` can instead capture another entry point,
including a validation loss, while preserving registered tensor names. The
callback is needed only at build time. See the
[forward-only example](../../examples/forward-only.md).

`preserve_buffers=()` optionally names registered buffers whose functional updates
contribute to outputs without being committed to model state. Each captured call
starts from the current buffer values, including when that form is repeated by
`instantiate_examples`. Temporary values and returned views use the usual
profiled, planned storage. Buffer aliases are preserved together; parameter-backed
buffers are rejected. This is an explicit low-level build option.

Distributed forward can update each worker's own replicated buffers. Parameter
and submitted-input mutations remain unsupported on that path.

## Saved implementations and compatibility

Finalization runs complete tasks with valid producer outputs for control inputs.
It records compiler/kernel choices and imports needed to restore them. Profiling
and execution use those choices; incompatible or unseen tuned specializations
are rejected. Numerical backend settings also belong to implementation/profile
identity; see [saved numerical settings](artifacts.md#numerical-backend-settings).
In particular, configured precision-cast emulation preserves native low-precision
sigmoid-backward rounding through compilation.

External kernel tuning and dispatch belong to explicit
[`OperatorPreparation`](../advanced/operator-preparation.md) adapters. Build does not
discover providers by model name. Adapters must save selections, validate their
source/ABI and restore dispatch without silently choosing new kernels. Dynamically
registered/native operations and transitive libraries need provider-specific
support. Shared [resources](../../architecture/resources.md) likewise need saved
descriptions and checked initialization. Model-owned collectives require all
participants during build; see [distributed setup](distributed.md).

Current compiled artifacts embed a **process-local CUDA ordinal**. Measurement
and execution must use the recorded binding: `device=1` cannot relocate code
compiled for `cuda:0`. Set device visibility before CUDA initialization so each
worker can see its assigned physical GPU at the compiled local ordinal. Logical
rank mapping is a separate choice.
