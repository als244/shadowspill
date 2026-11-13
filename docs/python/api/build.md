# Build and finalize task implementations


The optional `chunking` build policy selects axis chunking for effective stages;
see the [policy reference](frontend.md#axis-chunking-policies).

## Capture and finalization

The builders separate export, partition, compilation and lowering from task
timing and runtime pools:

```python
from shadowspill import build_forward

program = build_forward(
    model, example_args=(features,), example_kwargs={"mask": mask},
    partition="auto", chunking=None, target="cuda:0",
)
program.save("program.ssp")
```

The model may have CPU or meta parameters. Build uses a
representative GPU to finalize local CUDA task implementations. Model-owned
distributed operators require their participants and saved
[provider resource descriptions](../../architecture/resources.md). It accepts normal
positional and keyword arguments, including nested structures and specialized
non-tensor values. It records semantic module/parameter names, output storage
geometry, scheduled compiler-buffer lifetimes and explicit unresolved memory
requirements for provider calls. No persistent-state initialization or task
duration profiling occurs. Generated kernels are finalized by executing complete
tasks with valid example controls and outputs from their producer tasks. This
avoids benchmarking indexing kernels on unrelated synthetic integer buffers
during standalone compilation. Selected binaries are saved in the Program;
measurement and execution do not select new kernels or depend on an earlier
process's compiler cache.

The saved Inductor tuners also disable post-load reduction tile expansion,
coordinate-descent and combination searches, and lookup of a different cached
winner. Those controls apply to the saved copies after build finishes tuning;
ordinary build and eager autotuning keep their configured behavior. Retaining
one binary alone is insufficient because Inductor can otherwise introduce new
reduction candidates when loading it in a fresh process.

When building with PyTorch's `emulate_precision_casts=True`, compilation also
preserves native CUDA BF16/FP16 `sigmoid_backward` scalar rounding for matching
strided operands. The native kernel rounds the complement and first product;
the ordinary AOT decomposition uses FP32 arithmetic inside the operation.
ShadowSpill makes those rounds explicit before AOT decomposition so they can
remain in one fused pointwise kernel. CPU, other dtypes, and builds with
emulation disabled keep their existing compiler behavior. This does not change
the selected chunk reduction or stored-gradient precision.

The applied numerical contract is versioned in compilation/cache identity and
recorded as `task_metadata[key]["numerical_contracts"]`. Chunked and composed
tasks include their children's contracts. Changing the arithmetic requires a
new finalized implementation and matching profiles; planning and cold execution
use the saved implementation without rerunning this transformation.

Paired finalization and measurement retain only resolved backward inputs while
invoking backward; forward-only inputs and outputs are released first.
Before preparing a different task, they release idle allocator cache so its
inputs do not fragment large cached buffers from another task geometry. This
preparation is outside measured task invocations.
Finalization also resolves backend outputs whose device was provisional in
compiler metadata, such as attention RNG controls returned on CPU. Those values
have host lifetimes and no GPU allocation; conflicting observed storage locations
fail build before profiling or planning.
An optional `forward_fn(model, *args, **kwargs)` selects another model method or
computation without changing registered parameter names. For example, an
evaluation loss can use `forward_fn=lambda m, *a, **kw: m.loss(*a, **kw)` with
targets in `example_kwargs`. The default calls `model.forward`. The callback is
only needed during build; the saved Program contains the captured computation.

## Training builders

`build_forward_backward` adds an objective and persistent gradients without an
optimizer. It compiles gradient-creation and accumulation forms for each stage:

```python
from shadowspill import (
    build_forward_backward, StepDataOrdering, apply_profiles, plan_program,
)

program = build_forward_backward(
    model, objective=objective, example_args=(features,),
    example_kwargs={"targets": targets}, num_examples=8,
)
# After measurement or explicit profiles, choose traversal entirely on CPU:
plan = plan_program(
    apply_profiles(program, profiles),
    ordering=StepDataOrdering(depth=2, breadth=4),
    memory_budgets=memory_budgets, communication_info=communication_info,
)
```

The objective receives `objective(model, *args, **kwargs)` and returns a scalar
loss, optionally with metrics. `num_examples` fixes how many complete calls the
program includes; ordering is a planning choice. Changing it reuses compiled
forms and task profiles. Model mutations or RNG effects currently restrict
reordering. Tied parameter contributions preserve both creation and mutation
dependencies. `graph_pair_policy` defines independent saved-state cuts during build;
the planner separately chooses among permitted stage variants and decides
residency. See the [retention contract](frontend.md#graph-pair-policies)
for default classification, explicit rules and selective saved cuts.

For differing example shapes or static metadata, both training builders accept
`example_calls=[(args0, kwargs0), (args1, kwargs1)]` instead of
`example_args`/`example_kwargs`/`num_examples`. Matching signatures reuse captures;
each example retains its own signature and constants through save/load. The
default one-example-plus-count form remains sufficient for uniform inputs.

`build_step` takes the same training arguments plus `optimizer=optimizer`, whose
parameters must belong to the supplied model. It discovers optimizer tensor
shapes and initialization defaults without allocating persistent state or
changing that optimizer. Traceable local updates are packaged with the stages.
`plan_program(..., optimizer_ordering="stage_interleaved")` places updates after
their required gradient writes and all remaining reads of old weights;
`optimizer_ordering="tail"` puts updates after backward. Both use the same
Program and profiles. Opaque updates and unsupported first-use state creation
are currently rejected explicitly.

`master_dtype=torch.float32` captures separate optimizer master weights while
preserving the model's compute dtype. `grad_dtype` independently selects stored
gradient precision. The optimizer determines moment/state dtypes from its own
rules. Execution initializes masters from compute weights using bounded chunks;
exports/checkpoints name them `master.<parameter-name>`. The update contains the
gradient conversion and write-back to compute weights, so those costs and
allocations are profiled and planned. Build does not modify the supplied model
or optimizer.

## Saved implementations

Packaged tasks retain importable Python custom-op registration modules, their
source digests and operator schemas. Measurement and execution restore/check
these registrations before loading task code; the original model need not be
imported. Native or dynamically created registrations and cross-module kernel
overrides require separate provider support. Shared backend resources use the
optional, generic [`resources` builder argument](../../architecture/resources.md), with saved
descriptions and checked initialization code. Stateless-op tests alone do not
qualify these resources.
Likewise, the registration digest is not a fingerprint of all transitive native
libraries or proof that a provider's internal autotuning/resources are portable.

The PyTorch frontend records Triton compiler selections while finalizing each
unique executable with bounded representative inputs. Backward uses the
corresponding forward's tensors. Kernel source, tuning keys and candidate
configurations identify each selection and contribute to profile compatibility.
Measurement and execution bind these choices and reject unseen specializations
or changed candidates before benchmarking. Fixed kernels may compile lazily
without selecting a different implementation. This compiler integration does
not discover or patch external operation libraries.

External operator tuning and dispatch belong to the operation provider's
[`OperatorPreparation`](../operator-preparation.md) adapter. Its opaque manifest
can contain selected configurations together with immutable binaries. The
provider must observe warm dispatch hits, isolate task-owned host caches,
validate its source/ABI and reject missing specializations. ShadowSpill only
owns the generic capture/export/load/activate lifecycle, artifact integrity,
profile compatibility and physical memory admission. Ordinary calls outside a
provider's activation retain their ordinary behavior.

Callers explicitly supply the preparation descriptions for the operator
implementations their model uses; the builder does not discover libraries or
choose adapters by model name. Kernel export/load tests belong with the provider,
and compiler integration tests belong with the adapter. Core tests the generic
lifecycle and saved training through these public interfaces.


See [repository structure](../../development/repository.md) for implementation owners.

## Saved code and worker devices

The current Inductor artifact format embeds a CUDA ordinal. Build records it;
measurement and execution require the same process-local ordinal. In particular,
setting `device=1` does not relocate code compiled for `cuda:0`. Device visibility
can be arranged before CUDA initialization so separate workers each see their
assigned physical GPU at ordinal zero. Execution requires the recorded compiler
device binding. This limitation is separate from logical rank assignment.

Worker-group binding preserves explicit logical membership order
after applying the complete worker-to-logical-rank map and rejects inconsistent
maps/Plan identities before backend group creation. Group validation alone does
not establish provider correctness; each distributed backend needs qualification
through complete build, measurement, calibration and fresh-worker execution.
