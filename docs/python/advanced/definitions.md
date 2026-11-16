# Installed model and objective definitions

Register a custom model or objective through `shadowspill.training.definitions`.
Install the engine's `training` extra on the server to use these hooks. Portable
input and control schemas come from `shadowspill_client.protocol`. Start with
[basic training](../../examples/client-training.md) if you are using a supplied model.

## Model and objective hooks

A model definition describes an ordinary factory, validated configuration and
input schema. An optional `capture_context(config)` context manager scopes
provider precision or dispatch settings around construction and capture in the
build worker. It receives the resolved `ModelBuild.config` and must restore those
settings on exit, including failure. CPU description and training execution never
enter it; source and configuration are pinned with the capture. For example, an
MLOps model can use `weight_gradients_at(torch.float32)` without leaking that
setting into another model's capture. Gradient storage still uses
`ModelBuild.grad_dtype`; provider arithmetic and stored gradients are distinct.

An objective receives `(model, inputs, loss_inputs, loss_params)`
and returns the existing `ObjectiveResult`, a scalar loss or `(loss, metrics)`.
The objective owns its reduction. `simple_loss` adapts an ordinary
`loss(predictions, targets)` function without changing its reduction.
For a forward-only output set, a bound computation can return a mapping of its
declared tensor outputs and scalar metrics without computing a training loss.

Register both with a `Registry` using `add_model` and `add_objective`. Each
registration supplies a `DefinitionSource`: distribution name, version and
SHA-256 digest of the definition's source closure, including model and objective
dependencies. IDs contain an explicit version, for example
`my_project/image_classifier@1`. Re-registering an ID is an error. Lookups do not
import Python modules named by requests. Supplying an expected source rejects
changed definitions, including changes in editable installations.

An installed package can publish CPU registration hooks in its `pyproject.toml`:

```toml
[project.entry-points."shadowspill.definitions"]
models = "my_project.definitions:register"
```

`load_registry(["my-project-workloads"])` loads only the configured distributions,
calls each `register(registry)` hook in entry-point name order, and returns the
ordinary `Registry`. Hooks register in place and return `None`; they still supply
the source closures described above. Missing packages, duplicate package/entry
names and registration failures are errors. These package names belong to trusted
server configuration; model and compute requests continue to use definition IDs.
The supplied workloads wheel publishes the same entry-point group. This is the
common loader for controller and fresh-phase startup.

## Supplied definitions

The small [external classifier fixture](../../../packages/engine/tests/engine_tests/fixtures/external_definition.py)
shows the full interface. The supplied
[regression definition](../../../packages/workloads/src/shadowspill_workloads/definitions/regression.py)
uses the same registry and its existing model. Importing either description
requires no Torch import; model construction is explicit.

`shadowspill/llama3@1` supplies packed preparation, with BF16 compute and FP32
gradients by default. `Model.precision` accepts `compute_dtype`, `grad_dtype` and `lora_dtype`
(FP32 or BF16); optimizer masters are a separate optimizer setting. `config`
selects a named `preset` (default `tiny`) and optional `architecture` overrides.
LoRA uses the family's existing targets unless supplied explicitly; its current
provider supports zero dropout. Initialization supports explicit random weights,
pinned pretrained Hub weights and native training checkpoint resume.

The Llama batch declares `token_capacity`, `max_sequence_length` and
`max_sequences`. An optional `typical={"sequence_length": ...}` affects timing
samples only. The preset requests half-capacity and full-capacity buffers with
counts one and two; admission discards combinations exceeding the scheduled cap.
Explicit candidate lists replace those choices. Physical sequence slots include
separate padding segments, preserving caller boundaries even with empty Datums.
Every named output closure is packed and validated by the generic engine.

`shadowspill/qwen3moe@1` and `shadowspill/qwen35moe@1` use the same packed
preparation contract. They expose target logprobs and router statistics, so the
caller explicitly chooses token-only training or a composed balancing objective.
Settings select `preset`, architecture overrides, precision and optional LoRA;
Qwen3.5 recurrent decay parameters retain FP32 storage with BF16 projections.
Initialization accepts random, pinned pretrained or exported weights, and exact
training checkpoint resume.

BF16 MoE definitions request the existing provider preparation lifecycle and pin
Quack, SonicMoE and Cutlass sources. The provider selects supported kernels during
build and restores the saved choices during execution. Allocation contracts cover
ordinary unquantized BF16 Sonic calls and their prepared activation wrapper.
BF16 LoRA definitions also supply the provider's existing gradient-destination
rules and allocation contracts. The compiler fuses eligible complete, contiguous
gradients; unsupported crops or dtype transformations retain ordinary accumulation.


`shadowspill/olmoe@1` uses that same packed preparation, precision/LoRA settings
and bounded random initializer. Its model exposes the same router-statistics
capability; no OLMoE-specific loss is required. To retain OLMoE's layer aggregation,
configure the shared balancing term with `layer_reduction="sum"` and
`normalize_top_k=True`. Padding is excluded by default.

The three MoE definitions also describe `implementation="mlops_ep"` for the
existing Quack expert provider. Setup uses generic parameter/input placement,
BF16 compute and bounded random initialization. Each prepared physical token
size has a distinct communication resource; operations at the same size share
it, and parameter state remains common across sizes. These definitions support
BF16 full training with random initialization or checkpoint resume; EP LoRA and
pretrained initialization are unsupported.

Definition provenance hashes provider Python and installed extension binaries
without importing their implementations in the controller. Each controller or
worker treats that installed provider snapshot as immutable for its lifetime.

## Text objectives

The workloads entry point also installs `shadowspill/cross_entropy@1` and
`shadowspill/clipped_policy@1`. A compatible model declares the `target_logprobs`
capability, a bounded integer `tokens` field with one symbolic position dimension,
and a `target_logprobs(targets=..., **inputs)` method returning aligned target
logprobs. The model owns its fused or chunked head; the objective never asks it
to return all vocabulary logits. Other model inputs pass through their declared
schema and adapter. Vocabulary range, alignment and packing mode are derived at
setup, without callers repeating architecture settings.

Cross entropy takes aligned `target_tokens`, optional boolean `loss_mask`
(default true), and optional signed `weights` (default one). Clipped policy
requires `target_tokens`, boolean `response_mask`, `old_logprobs` and `advantages`,
with the same optional weights. Its call-local `clip_ratio` defaults to 0.2 and
must lie in `[0, 1)`. Preparing `config={"reference_penalty": "k3"}` additionally
requires `reference_logprobs` and enables nonnegative `reference_beta` (default
zero). Without that prepared term, reference fields and controls are rejected.
These are FP32 weighted sums, with no implicit division. A zero weight or
advantage still counts as a valid token.

Both objectives provide `metrics` (`loss_sum`, `valid_tokens`) and `logprobs`
output sets. Training with `logprobs` also reports those metrics, with zero tensor
outputs at masked positions. `forward(..., outputs="logprobs")` needs only model
inputs and aligned targets; it returns every target logprob without requiring
rollout fields or loss controls. Forward requests for `metrics` evaluate the
declared loss and therefore require its fields. Omitted controls always revert
to setup defaults.

`resolve_model` and `resolve_objective` validate configuration and return immutable
portable settings, input schemas and provenance. Configuration factories return
dataclasses or mappings. The resolved model's `create()` is a separate preparation
action. Objectives declare required model capabilities; resolution rejects
incompatible bindings. An objective with several named output sets supplies a
`bind(config, OperationKey)` function selecting the computation for that operation,
output set and model mode. `OutputSet(forward=...)` can describe a separate
forward-only contract under the same output name. For example, policy training
needs advantages and old logprobs while a logprob-only forward does not. Both
contracts are inspectable and saved; preparation selects the exact one before
validation, packing and capture. A forward in training mode still uses the
forward-only contract.

Objectives may supply `configure(values, resolved_model)` to infer structural
settings from the model's CPU description. The normal typed configuration factory
checks its result; resolving those defaults again must give the same settings.
Like input schemas, controls and output mappings can be functions of that typed
configuration. These functions run during setup, never during runtime requests.
The resulting concrete contracts are validated and pinned in the saved setup.

`Objective.sum({"main": main_objective, "regularizer": other_objective},
weights={"regularizer": 0.01})` combines named installed objectives. Coefficients
default to one; per-call `loss_params={"regularizer.weight": 0.02}` overrides one
coefficient without changing later calls. Child controls and results are prefixed
by their term name, for example `main.clip_ratio` or `main.loss_sum`. Training
returns the weighted total as `loss_sum`; each child metric retains its original,
unweighted meaning. `grad_scale` still applies once to the accumulated gradient.
Zero coefficients do not change participation or skip evaluation of a term.

To compose, an objective declares `evaluate(config, key) -> ObjectiveEvaluation`.
Its `compute`/`bind` reducer then receives evaluated values as its first argument.
The evaluation record names a versioned interface, its installed callable, and
portable `requests`. The callable takes `(model, inputs, loss_inputs, requests)`.
All terms must share that interface and callable. Setup merges their requests,
rejects conflicting values, and execution evaluates the model once before running
the reducers. Reducers must not mutate shared values. An ordinary model-aware
objective without this hook still works alone; composition requires an explicit
hook rather than automatically repeating or memoizing model calls.

Terms share loss-input names and must agree on overlapping field schemas,
capacities and the participation field. Use distinct names for distinct inputs.
`weight` is reserved as a child control. Nested sums must be flattened into named
terms. The resolved setup pins every child source; changing any child invalidates
the saved composition. Custom installed objectives use this engine feature with
no catalog dependency, and constructing the settings needs only the client.

The sum's `metrics` output selects each term's default. Other output names are
the union of the terms' additional output sets. Training selects that set where
available and the default for other terms, so requesting tensors retains every
loss term. An output-only `forward` selects only terms declaring that output;
it needs their declared forward fields/controls and returns their prefixed
outputs. Forward returns per-term metrics without a total training loss or
requiring composition coefficients.

Models that need geometry-dependent construction or provider resources can supply
`build(config, ModelBuildContext)`. This CPU hook returns `ModelBuild` with complete
constructor settings, existing `Resource` descriptions and optional
`OperatorPreparation` descriptions. Its context includes the current logical input
layout, parallelism and the complete requested layout family, so a shared provider
buffer can cover all shapes. The same model factory receives the normalized
settings during capture. Resource creation, binary capture, restoration and cleanup
remain in the existing phase engine. Description never opens a provider or imports
a factory named by a compute request; definitions own those source-pinned choices.
Templates save and recheck the resolved construction and actual provider inventory.

`ModelBuild(grad_dtype="float32", ...)` selects the existing engine's captured
gradient storage and accumulation precision. This differs from parameter or
operator-internal computation precision. `OptimizerDefinition.master_dtype` is
an optional floating dtype name or CPU function of the resolved optimizer config;
it enables the existing master-parameter capture. Both choices enter capture
identity. The ordinary builder retains one compute/optimizer/gradient inventory;
backward and optimizer-only Plans derive from the same step capture.

`ModelDefinition.capture_context(config)` optionally scopes provider precision or
dispatch during capture. An installed `gradient_destination_rules(config)` factory
may supply the existing `GradientDestinationRule` contracts for backward operators
that write into accumulated gradients. Both hooks receive the normalized build
configuration. CPU inspection never calls them; forward-only capture does not
request gradient rules. The ordinary compiler validates and lowers those rules,
and saved workers execute the resulting operations without calling the factory.
Opaque replacements still require their exact allocation contracts. These are
model integration hooks; runtime clients keep submitting settings and inputs.

An installed `ModelReferenceDefinition` resolves pretrained metadata on CPU and
pins a model definition and immutable weight revision. It may fill missing fields
inside nested configuration or precision mappings. Every explicit value remains
fixed, including nulls and complete sequences; changed or removed values are
rejected. The same resolved settings are rechecked when creating a training run.

The supplied `shadowspill/huggingface@1` resolver accepts Hub repository IDs for
Llama 3, Qwen3 MoE, the Qwen3.5 MoE text decoder and OLMoE. It reads bounded
config/index metadata and shard sizes/checksums, pins the immutable Hub commit and
uses the existing catalog parsers. Tensor payloads are fetched only after worker
resources are reserved.
The resolved manifest belongs to `Weights.metadata`, separate from model
configuration. Selecting another compatible weight source or fresh random weights
does not change the model computation. Explicit pinned metadata is rechecked and
cannot be silently replaced. Clients and servers use protocol version 2; older
versions are rejected explicitly.
It imports no tensor framework and executes no repository Python. An explicit
`config={"architecture": {"max_seq_len": ...}}` replaces the metadata's context
capacity; overrides that change stored tensor geometry are rejected. Other model
settings, including explicit precision and LoRA targets, remain caller-owned.
Installed compute/gradient precision defaults apply independently of the stored
weight dtype. For Hub references, the server's ordinary Hub authentication and
endpoint settings apply, and the client supplies a repository ID.

The supplied `shadowspill/local_safetensors@1` resolver also accepts an absolute
server path through the same `Model.from_pretrained("/checkpoints/model", ...)`
call. It supports those same four families and checkpoint overrides. First
resolution fingerprints the files with bounded SHA256 reads; this scans the
checkpoint payload once without constructing tensors. Reusing the resolved setup
rechecks bounded config/index/header metadata without repeating that scan.
Workers verify all content checksums before and after initialization. The source
must stay unchanged while it is read. A private temporary directory exposes only
the pinned files through links, then is removed; the source checkpoint is neither
modified nor copied. Local loading uses the same bounded readers and initializer
limits as Hub loading. Paths are interpreted on the server, never on the laptop.

The supplied pretrained loader verifies content checksums, streams downloads in
1 MiB chunks and uses the existing parameter-range readers. Compatible cached Hub
files are read without modification. New files live in a private temporary directory
and are removed when the loader closes, including on failure; a later restore can
fetch the same pinned source again. Config/index/aggregate-header bounds are
1/4/4 MiB, with at most 512 shards and 65,536 tensor declarations. The declared host
allowance covers metadata decoding, retained reader state, conversion and generated
buffers; full model tensors are never assembled by the loader. Derived rotary
buffers and fresh LoRA factors use the existing deterministic initializer. Fresh
adapters default to seed zero for referenced weights, or the base seed for random
weights. Set `Model(..., config={"lora_seed": 45}, lora=LoRA(...))` (also accepted
by `Model.from_pretrained`) to choose their seed independently. This setting
changes only newly generated adapter factors; base weights and rotary buffers
are unchanged. Native resumable checkpoints preserve trained adapters and
optimizer state through the engine's checkpoint path.

Fresh worker initialization uses an optional
`initialization(config, InitializationContext)` hook. It describes the pinned
`Weights`, complete model settings, parallelism and logical rank on CPU, without
constructing a model, importing Torch, opening weight files or downloading data.
Return `ModelInitialization(peak_bytes=..., chunk_bytes=..., open=...)`: the peak
includes retained loader state, all CPU scratch and the maximum destination chunk,
separate from the admitted fixed model pools. Creation checks this bound against
host staging before launching workers. Inspection and preparation may omit initial
weights; fresh creation requires explicit weights and a bounded installed loader,
or a compatible checkpoint restore.

An exported weights generation omits nonpersistent model buffers, such as derived
position tables. Models that need these values declare
`temporary_initialization(config, InitializationContext)`, returning the same
bounded `ModelInitialization` description. `context.temporary` names the exact
buffers to generate; the worker restricts the callback to those names. This hook
uses model settings and opens no historical weight source. Saved model values and
fresh optimizer defaults are initialized separately. Models without such buffers
need no additional hook. A partial export can reuse its declared frozen-base
initializer for these derived values.

An initializer that downloads or extracts files also declares `disk_bytes`
(default zero). The node reserves the sum for its workers before opening sources
or launching children. Workers share a private initialization cache within that
reservation; it survives for frozen-base restores and is removed after workers
stop. Cleanup releases the disk claim only after removing those owned files.
Existing user caches can be read without modifying or deleting them.

The worker reserves the peak before entering `open(InitializationRuntime)`.
That context manager yields `fill(TensorChunk, flat_cpu_destination)` and releases
its resources before the reservation ends, including on failure. The existing
Executor visits only owned tensor regions and handles captured defaults,
optimizer state, master copies and tied storage. The loader fills the requested
storage range, using the descriptor's original geometry and offset; it must not
construct an unbounded full model or invent values for unrecognized parameters.
Random loaders can reuse `checkpoint.initialization.fill_generated`, whose fixed
per-name blocks preserve values across chunk sizes and rank ownership. The
external classifier fixture demonstrates this path.

Pretrained loaders require a pinned revision. Artifact initialization uses an
opaque, owner-granted ID resolved by `InitializationRuntime.artifact`; a request
never supplies a server filesystem path. Runtime cache paths and loader callbacks
are worker-local and never enter saved Plans or public wire records. Loader code
and its dependencies belong to the model definition's source closure. Worker
checkpoints record that source, the resolved settings and exact weight reference
when omitting unchanged frozen parameters. The loader must reproduce those weights
independently of chunk order and rank ownership. On partial restore it receives
only omitted parameters; trainable state, mutations and buffers come from the
checkpoint. The worker admits the retained loader peak together with restore
scratch before opening it. Full checkpoints require no base loader.

Tensor wrappers may describe stored component coordinates through
`ModelBuild(component_layouts=[ComponentLayout(...)])`, importing `ComponentLayout`
from `shadowspill.training.definitions`. A declaration identifies the exact
parameter, its structural `component_path`, the parameter axes used by that
component, and a versioned `encoding` identity. For example,
`ComponentLayout("layer.weight", ("payload",), (0, 1), "custom/bf16@1")` declares
that the component uses the same two coordinate axes as its logical parameter.
Capture checks its shape against the registered component; saved Plans retain and
recheck the declaration. Existing bounded checkpoint and weights readers then
redistribute the physical payload alongside ordinary masters and optimizer state.
The encoding and axis interpretation must agree across source and target layouts.
No values are decoded or requantized by this declaration. Packed dimensions with
different extents require a separate conversion; matching shapes alone never
authorize a component mapping. Ordinary tensors need no declaration.

Opaque kernels used by prepared operations need an all-value allocation contract.
`ModelBuild`, `ObjectiveDefinition` and `OptimizerDefinition` accept
`allocation_invariants`, mapping exact compiled call names to versioned contract
IDs, for example `{"torch.ops.my_provider.scale.default": "my_provider/scale@1"}`.
The installed author guarantees identical allocation geometry, output aliases and
retained storage for every tensor value and reachable provider state at the
captured geometry, resources and non-tensor arguments. Tensor controls are included;
hidden or asynchronous allocations are excluded. An operator-preparation factory
also needs an exact `module:factory` entry covering bound-task activation/wrapping.
These are source-pinned semantic guarantees, like padding safety; sampled peaks
cannot supply them. Wildcards, conflicting declarations and unresolved compiler
geometry are rejected. Value-dependent allocation needs a separate maximum-bound
mechanism and currently cannot qualify a prepared family through this contract.

Each `TensorField` describes one path in a Datum's input or loss-input tree:
dtype, per-example shape, padding and optional default, and stacking, packing or
shared-value semantics. Repeated symbolic dimensions explicitly declare aligned
positions. Matching integer sizes alone carry no alignment meaning. Capacities
declare their units, counted field axis and global sum/maximum or example count.
Scheduled capacities include padding. These descriptions do not establish
packing coverage or admit memory; that work belongs to preparation.

An optional model `input_adapter(config, layout)` returns fixed views and bounded
generated metadata for the selected physical buffer. The ordinary independent,
fixed stacked case needs no custom adapter. Packed, variable or grouped model
inputs require an explicit adapter, including sequence/group boundaries when
needed. The adapter declares `padding_safe=True`: padding preserves real model
outputs. The objective normally declares the same guarantee, with padding
contributing no loss or scalar totals. A validity field alone does not establish
this guarantee. A custom objective that deliberately uses physical microbatch
statistics instead declares `physical_batch_loss=True`, `reduction="custom"`
and `padding_safe=False`. Its loss and gradients can change with the selected
buffer size, padding and split calls. Preparation records this distinction; it
does not change batching, capacity limits or all-masked update skipping.

`shadowspill/moe_balance@1` supplies reusable softmax router balancing. Compose
it with cross entropy, clipped policy or another compatible token objective:

```python
objective = Objective.sum(
    {
        "tokens": Objective("shadowspill/cross_entropy@1"),
        "balance": Objective("shadowspill/moe_balance@1"),
    },
    weights={"balance": 0.001},
)
```

The default excludes artificial padding. Its optional aligned `router_mask`
defaults to true for real input positions and false for padding, independently
of the prediction mask: real prompt tokens still contribute routing statistics.
`config={"include_padding": True}` explicitly counts all physical rows.
The buffer's full allocated size still counts against the scheduled token cap.
The balancing value is scaled by the supervised-token count; primary token
weights do not scale balancing. The caller chooses its coefficient through
`balance.weight`. All-masked prediction supervision still skips the update.

For a composed objective, `forward(..., outputs="metrics")` returns the weighted
loss when every child returns a scalar loss. Its namespaced coefficients follow
the same setup-default and call-local override rules as training. If a child
provides only outputs, the result contains the declared child metrics and no
partial combined loss. Prediction-only output sets need no loss coefficients.

The default `layer_reduction="combined"` combines counts/probability sums across
layers before computing balancing, with `normalize_top_k=False`, matching Qwen's
aggregation. `layer_reduction="sum", normalize_top_k=True` uses OLMoE's existing
sum of per-layer terms, dividing assignment frequency by top-k. `"mean"` averages
per-layer terms. With clipped policy, set `participation_field="response_mask"`;
otherwise it defaults to `"loss_mask"`. Padding exclusion does not make this
microbatch statistic additive: whole and split calls need their own oracles.

A compatible model declares `router_statistics` and implements
`token_outputs(targets=None, router_mask=None, **inputs)`. It returns per-layer
`router_counts` and differentiable full-softmax `router_probability_sum` tensors
of shape `[layers, experts]`, scalar `router_rows` (selected positions per layer),
integer `router_top_k`, and aligned `logprobs` when targets are supplied. A null
router mask explicitly means all physical rows. Qwen and OLMoE supply this hook.
Compatible primary and secondary terms share one transformer evaluation. The
current fused provider lacks masked statistics, so exclusion adds a bounded router
projection/selection per layer; its work and storage must be captured and admitted.

The input pipeline validates owned binary snapshots with these same
schemas before packing. It checks every supplied field, shared dimension, numeric
range and global capacity, including later records in the call. Required paths
come from the selected computation/output closure; known unused fields may be
supplied, but are still checked. Optional defaults retain their broadcast shape
without allocating a second full batch. Checked dtype conversion happens when
filling an admitted packed buffer. CPU validation handles generic image geometry
and token fields through the same implementation, without importing a model stack.
Packing still must check scheduled padding against the cap separately.

A field may name an identically aligned boolean `validity` field in the same
tree. When that mask belongs to the selected operation's input closure, numeric
checks apply to active positions; inactive values are replaced with the declared
safe padding before dtype conversion. This permits masked sentinel targets and
NaNs in unused auxiliary positions without propagating them into loss arithmetic.
Validation and conversion use bounded chunks. If the operation does not select
the mask, every position is checked normally: submitting an unused training mask
cannot hide an invalid target from a logprob-only forward. Shape checks always
apply, and invalid values at active positions reject the complete call.

`Participation` identifies a boolean validity field with false padding, or
declares that each real example contributes. It does not inspect weights,
advantages, observed gradients or gradient scaling. `ControlSchema` validates
finite scalar controls and resolves setup defaults plus call-local overrides
without retaining overrides. Output sets declare needed fields/controls, tensor
ordering and scalar reductions; unknown fields and capabilities are errors.

## Preparation and capture

The internal operation binder selects only the requested input/control closure,
checks returned tensor shapes/dtypes and scalar metrics, and applies declared
logical loss ownership without averaging. It saves definition and layout identity
with each Program. Capture admission verifies actual coefficient reads and rejects
input mutations, undeclared model-buffer mutations and any model-state mutation
in evaluation. The existing builders produce optimizer, backward and output-only
paths; optimizer builds enable the captured final gradient scale.

Installed `PreparationDefinition` functions provide CPU defaults and physical
buffer geometry for each logical input partition. Register them with the same
source-pinned registry; a model can name its default with `preparation`. The
engine supplies `shadowspill/stacked@1` for independent fixed stacked inputs.
Variable, shared or coupled inputs require an appropriate installed preset.
The preset's size units belong to its input contract. Explicit candidate lists
replace inherited lists, fixed `chunking=None` remains unchunked, and complete
cases preserve coupled choices. A coupled-case preset requires complete case
overrides when changing its candidate axes.

The internal CPU resolver emits a portable `ResolvedPreparation` containing
complete build cases, profiling/planning settings, source identity and each
choice's origin. Omitted planning workers inherit the preset and then the engine
default of zero; explicit zero selects all available logical CPUs. Existing core
`SearchOptions` remain the search description. Expansion rejects schedules whose
padding exceeds hard limits before repeating buffers, and reports case-limit
errors without shrinking the request. Geometry coverage is checked separately
for every requested batch option.

`training.preparation.setup.resolve_training` resolves the complete portable
setup and rechecks saved configurations on CPU. `prepare_layouts` in the adjacent
`layouts` module binds each operation, physical buffer and logical worker to its
input adapter and output geometry, then checks the complete host-staging bound.
Known staging-limit failures retain candidate reasons; unexpected definition
errors propagate. Every requested operation and batch option must retain a
constructive coverage witness. Identical adapter layouts share their resolved
metadata across counts.

The internal `training.preparation.bundle.TrainingBundle` saves a complete admitted
operation family through the existing phase-archive codec. It retains the resolved
setup, referenced source captures/profiles, per-worker Plans, input/output layouts,
coverage witnesses and shared-state admission. It contains no live weights or
optimizer progress. `TrainingBundle.from_prepared(family, sources).save(path)`
accepts the result of `plan_family` and its original measured source profiles.

`TrainingBundle.load(path)` decodes an inspectable CPU artifact. Before creating
workers, call `bundle.recheck(registry)` to validate installed sources and exact
capabilities, re-derive operation/count Programs from saved captures, replay saved
schedules and recheck geometry, allocation coverage and shared admission. Saved
allocation coverage binds the complete capture and every compiled form, including
additive gradients, to static compiler geometry and exact opaque-call contracts.
Measured traces still supply byte sizes and physical placement. Missing masked/optimizer or
accumulation paths are errors. Loading does not capture, profile, search, select
replacement candidates or initialize training state.

`training.preparation.job.prepare_job(directory, packages=..., config=...)` runs
resolved preparation through the existing fresh-process build, profile,
communication and CPU planning phases. The caller supplies an assigned worker set
and an owned job directory. The job saves its exact configuration, rejects
concurrent callers, reuses compatible completed captures/profiles, and retains
recoverable candidate failures only while all requested capabilities remain covered.
A completed job reloads and strictly checks its bundle without starting phases.
It never initializes weights or starts a training loop.

Phase policies use `PolicyDefinition(id, kind, config, create)`. The CPU resolver
validates and fills `Policy` configuration through the registered factory's
configuration schema, records its source, and rejects unknown policies or a
wrong phase kind. Reusing the resolved settings rechecks both source and exact
defaults. Only the build boundary calls `create`, which returns the ordinary
partition, chunking or graph-pair policy consumed by the existing phase engine.
Request strings never name modules to import. The built-in partition definition
supports the engine's `auto` and `whole` choices.

The workloads package registers `shadowspill/text_chunks@1` and
`shadowspill/text_retention@1` around the existing unquantized text policies.
For packed inputs, `token_capacity` in the chunk policy is the physical buffer
width. Keep it paired with the corresponding microbatch size using complete cases:

```python
from shadowspill_client import BuildCandidates, BuildCase, BuildOptions, Policy, Preparation

prepare = Preparation(
    candidates=BuildCandidates(cases=[
        BuildCase(size, BuildOptions(
            chunking=Policy("shadowspill/text_chunks@1", {
                "token_capacity": size,
                "chunk_size": 4096,
                "expert_chunk_size": 8192,
            }),
            graph_pair_policy=Policy("shadowspill/text_retention@1", {
                "name": "cost-input", "chunked": True,
            }),
        ))
        for size in (8192, 32768)
    ]),
)
```

The expert chunk size defaults to the ordinary chunk size. Retention supports
`cost-input`, `cost-attention` and `attention-input`; `chunked=True` preserves the
existing provider-local scratch rules. These settings affect internal execution,
not logical update boundaries, objective normalization or the scheduled token cap.
Factories remain in build workers; setup inspection loads only descriptions and
source identities. Saved Plans restore the selected execution without recreating
these policies on each training call.

## Optimizer definitions

`OptimizerDefinition` registers a factory for an ordinary capturable optimizer.
The factory receives `(model, config, controls)`, where the build worker provides
tensor-valued runtime controls. Each advertised control explicitly names its
captured scalar bindings. After existing `capture_optimizer` runs,
`validate_capture` checks that those bindings are read by the captured update.
Static options and unused tensors cannot be advertised as runtime overrides.
The registry resolves nonsticky defaults before construction; it does not run
optimizer steps or introduce another optimizer implementation.

The engine distribution publishes an optional optimizer registration hook. Include
`"shadowspill"` in the server's `workload_packages` alongside your installed model
package to enable `Optimizer("mlops/adamw@1")`. This requires MLOps on the server;
external packages can supply their own optimizers without loading this hook.
Registration hashes the installed provider source without importing Torch or
constructing an optimizer. Construction calls the existing `mlops.optim.AdamW`.

This definition currently supports one parameter group, selecting only trainable
parameters. Runtime controls are `lr`, `beta1`, `beta2`, `eps` and `weight_decay`;
omitted values use the immutable setup defaults. Structural `config` supports
`maximize`. `precision` accepts `gradient_dtype` and `opt_state_dtype` (`parameter`,
`float32`, `float16` or `bfloat16`), plus `parameter_rounding` and
`opt_state_rounding` (`nearest` or `stochastic`). Defaults match MLOps: BF16 update
gradients and moments, nearest rounding. These dtype settings describe the
optimizer's input conversion and moment storage; they do not change model or
gradient-accumulation storage. An additional `master_dtype="float32"` option
selects separate FP32 optimizer parameters through the existing build engine;
its default is `None` (no master copy). Unsupported groups and precision
options fail during description. For the existing FP32 regression recipe, specify
`precision={"opt_state_dtype": "float32"}` to preserve its optimizer settings.

The supplied `shadowspill/regression@1` definition accepts explicit
`Weights.random(seed=...)` initialization and compatible checkpoints. Seed zero
reproduces the original regression recipe's initialization, independently of
worker ownership and loader chunk sizes. Its `predictions` output needs only
`Datum.inputs["value"]` (16 FP32 values), and evaluation leaves the model's
training counter unchanged. `shadowspill/weighted_squared_error@1` takes eight
FP32 targets per Datum, optional scalar `weights` and boolean `valid`, and returns
the weighted squared-error sum. To reproduce the original global mean, supply
`weights=1 / (8 * number_of_global_examples)`; the engine adds no normalization.

The phase engine also provides `artifacts.state_inventory.program_state(program,
logical_rank=0)`. It describes captured parameters, buffers, masters, persistent
gradients, optimizer tensors and host counters using role/name/component IDs.
Saved geometry, parameter ownership and physical storage bindings remain distinct.
The inventory serializes without importing Torch or the SDK. It detects layout,
optimizer-family and alias conflicts when matching state between Programs, and
refuses to alias different logical workers. This is the shared description for
shared mutable Executor Plans and upcoming checkpoint conversion. The current
public Trainer still accepts its existing step/evaluation interface; the client
service integration is under development.
