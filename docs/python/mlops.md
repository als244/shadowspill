# ShadowSpill integration

`shadowspill.integrations.mlops` supplies explicit provider contracts for
ShadowSpill's generic build controls. The integration belongs to this repository. MLOps exposes standalone declarations through `mlops.contracts`; this adapter
converts them to ShadowSpill build inputs. Neither declaration nor conversion
initializes CUDA, selects a model, or installs a process-wide policy.

```python
from shadowspill.integrations.mlops import (
    sonicmoe_chunk_rules,
    sonicmoe_gradient_destination_rules,
)
from shadowspill import build_step

program = build_step(
    model,
    objective=objective,
    optimizer=optimizer,
    example_args=example_args,
    gradient_destination_rules=sonicmoe_gradient_destination_rules(),
    chunking=chunking_policy,
    graph_pair_policy=graph_pair_policy,
)
```

The model must already use the SonicMoE LoRA provider. The destination factory
registers accumulating dispatcher overloads and returns immutable build rules;
it does not change provider dispatch. A build containing no matching operators
simply does not apply them. Pass the same rules in shared `build_options` when
searching chunk sizes and retention candidates. Each resulting task still needs
normal finalization, measurement and physical memory admission.

## Gradient storage and precision

The first parameter contribution allocates its gradient. Later eligible uses
across chunks, execution/checkpoint regions and microbatches use the supplied
destination. A single backward call handles all selected parameter outputs,
using one GEMM per product with `C` and `D` referring to that destination. It
does not create a separate product tensor and then add it. Empty experts preserve
the prior contents. Activation and routing gradients and other unconsumed outputs
retain their ordinary contracts.

Version 1 supports complete aligned contiguous FP32, BF16 or FP16 destinations
at the original gradient dtype, one finite literal scale, and canonical up-B
gate/up ordering. Narrow accumulation obeys ShadowSpill's explicit
`fuse_narrow_gradient_accumulation` permission: products, scale and prior sum
use the opmath epilogue with one final cast. With that permission disabled,
narrow destinations keep ordinary accumulation. Intermediate narrowing into a
wider final gradient, cropped/padded logical ranks, unknown transformations and
FP8 base containers also keep the ordinary path; casts are never removed merely
to enable fusion. One unsupported output does not prevent another eligible
output of the same backward from using its destination.

The dtype list describes the destination contract, not an expanded guarantee
about every Sonic kernel or end-to-end training dtype. Actual BF16-factor
training and two-rank data parallelism have dedicated integration coverage;
FP16 epilogue coverage alone is not FP16 model qualification.

## Chunk axes and state

`operator_chunk_rules(batch_size=1)` supplies the common normalization, SwiGLU,
rotary, gated-norm, basic mHC and builtin LoRA routing/dispatch/grouped-linear
axis/reduction contracts. These declarations live beside the provider, rather
than in model code. The batch factor describes flattened batch/position layouts;
it does not choose an independent axis or certify arbitrary feature splitting.
Pass the same batch geometry as the enclosing region. The factory's operator
names, semantic versions and rules are unchanged by the ownership move.

Combine those rules with the explicit SonicMoE and fused-mHC bundles when used.
Models still express full tensors; ShadowSpill generates execution chunk loops.
The workload chooses boundaries, domains, chunk sizes and retention policies.


`sonicmoe_chunk_rules()` returns seven `OperatorRule` objects for routing
metadata and the fresh/accumulating up/down operations. Add these to the rules
for other operators in each selected `AxisChunks` region:

```python
rules = (*other_operator_rules, *sonicmoe_chunk_rules())
# Use rules=rules in the AxisChunks values returned by chunking_policy.
```

The input domain and axis come from the caller's chunk policy. Sonic's sorted
rows and routing maps are chunk-local and must not escape a chunk; expert counts
and parameter gradients are additive reductions. An invalid saved-state cut
that exposes those local values must be changed or refused by the builder.
This is a representation constraint: saving the local values would require
separate per-chunk buffers and routing metadata. A selective SAVE policy may
instead regenerate these intermediates while retaining other state, including
the selected expert IDs.
These rules do not select checkpoint boundaries, retention, replay, chunk sizes
or spill residency. They do not certify expert-parallel collectives or arbitrary
provider effects as independent across an axis.

`sonicmoe_chunk_rules(compute_precision="fp8_block")` explicitly selects the
frozen block-FP8 expert contract, with BF16 activations/LoRA factors. It returns
five rules for the original operators, leaving the FP8 data/scales shared.
Activation quantization reduces features within each row, so token splitting
preserves its scaling domain. The FP8 contract has a distinct executable
identity and excludes destination fusion, trainable base weights and
`fp8_current`; ordinary in-place addition of gradient contributions remains.
Full/tail chunks, arbitrary VJPs, rank32 factors and empty experts have direct
H100 comparisons. Full-model saved execution remains a separate qualification.

`mhc_chunk_rules(batch_size=1)` supplies token contracts for fused DeepGEMM
preprocessing and TileKernels postprocessing. Pre expects packed
`[batch * sequence, 4, hidden]` streams; every invocation including tails needs
32-token/128-hidden alignment. Fake capture checks this geometry too. Shared
projection/bias/scale gradients sum across chunks. Post preserves its reshaped
`[1, tokens, ...]` inputs. These rules do not permit splitting the stream or
feature dimensions.

## FLA value lifetimes

`fla_execution_lifetimes(rules=(), select=None)` supplies readonly and input-layout
contracts for the explicit FLA convolution, L2/gated normalization and Gated
DeltaNet operations. The factory makes no scheduling choices on its own:

```python
from shadowspill.integrations.mlops import fla_execution_lifetimes
from shadowspill.pytorch import LifetimeRule

lifetimes = fla_execution_lifetimes(rules=(
    LifetimeRule("aten.slice.Tensor", compact=True),
    LifetimeRule("mlops.linear_attention_fla_gated_delta_rule_bwd.default",
                 barrier=True),
))
# Supply lifetimes on the outer ChunkRegions policy.
# Add movement rules or a selector for the rest of the desired schedule.
```

The engine still proves that copied internal views preserve aliases and logical
values, shrink a protected backing set and do not increase any other protected
set. Certified effects also let independently selected readonly work, such as
gradient padding, move past these operators toward its consumer. No projection,
expert or attention replay is selected. This does not make recurrent token
windows independent or change checkpoint retention/spill placement.

Only the listed value operands have layout permission. In particular, L2
backward accepts relayout of its incoming gradient, but its saved normalized
output and statistics retain their original layouts. Weights, sequence metadata,
other providers and ordered mutations are not given new permissions. The
factory is CPU-only and does not import FLA kernels. Resolved contracts and
choices enter executable identities; rebuild/profile changed policies before
planning and saved execution.

## Saved execution and validation

Mapper callbacks run during graph construction. Saved Programs carry the actual
replacement overload, its arguments/version and implementation dependency;
executors need the provider implementation but do not call these factories or
mappers. Normal graph compatibility, task profiling and PressureFit admission
still apply. Full/tail sizes can require distinct kernel specializations;
profiling measures the complete composed task that invokes them. This adapter
does not bundle external Triton/CuTe binaries or guarantee
that first invocation performs no external JIT compilation.

## Preparing Sonic kernels

The separate preparation adapter packages the selected BF16 Sonic/Quack
implementations with a saved Program. It is an explicit build option:

```python
from shadowspill.integrations.mlops import sonicmoe_preparation

prepared_sonic = sonicmoe_preparation()
# Add operator_preparation=(prepared_sonic,) to build_forward,
# build_forward_backward, build_step, or shared search build_options.
```

Constructing this description does not initialize CUDA. The builder opens its
factory before graph compilation, then captures selected external kernels in
each task's existing finalization call. Autotuner benchmark candidates are
excluded; warm winners, generated epilogues, tails and accumulation forms are
included. The task is not executed an additional time to collect binaries.

The adapter packages Quack objects and selected autotuner configurations,
external Triton CUBINs, the native launcher extension and the occupancy helper.
Quack-specific source checks and host-cache isolation are owned entirely by
MLOps; ShadowSpill treats this manifest as opaque. Saved profiling and execution bind these
assets without compiling or tuning. Loading validates implementation sources,
GPU target, CPU/host ABI, runtime contracts and binary digests. Missing or
incompatible implementations refuse; eager caches cannot supply a fallback.
The optional `cache_root` option controls build intermediates only. By default
these use `$XDG_CACHE_HOME/mlops/prepared` (or `~/.cache/mlops/prepared`). Set
`XDG_CACHE_HOME` to node-local storage in scheduled jobs. Restoring archived
binaries does not require that directory to exist.

Profiling loads before task measurements. Execution installs the engine's
allocator first, reserves the profiled external footprint, then loads selected
modules before Plan layout admission and the final physical-budget check.
Prepared libraries are shared across compatible task bindings and explicitly
released after streams drain. Binding activation does not load these modules.
The native Python launcher extension has separately validated process lifetime.
Prebuilt PyTorch/cuBLAS libraries may still initialize on first use. Their
external footprint remains reserved by profiling/admission; preparation does
not eliminate every source of first-invocation latency.

Preparation is independent of destination rules, chunk axes/sizes, checkpoint
cuts and save/recompute policies. Each search candidate still builds, profiles
and plans its own selected implementations. No precision settings, routing
choices, retention decisions or gradient arithmetic change.

The initial adapter covers the BF16 Sonic path. FP8, expert parallelism and
unrelated attention providers require their own preparation contracts. A small
real BF16 saved-Plan fixture has independent gradient/checkpoint coverage;
prepared full-model, other-dtype and distributed qualification remain separate.
The provider mechanisms and lifecycle limits are described in
[operator preparation](../operator-preparation.md).

CPU tests in `tests/workloads/test_mlops_contracts.py` exercise actual custom-op capture,
grouped outputs, dtype permissions, canonical layouts and conservative fallbacks.
`test_sonicmoe_gradient_destinations.py` covers schemas, functionalization and
ownership. `test_sonicmoe_gradient_destinations_gpu.py` checks real GEMM products,
same-buffer epilogues and empty experts against independent CPU FP64 products.
Its 12 product cases cover four layouts in FP32/BF16/FP16. Eight backward cases
cover full destinations in both directions and FP32/BF16, plus pairwise partial
LoRA destinations; they check each product runs once and unconsumed outputs
remain valid. These GPU checks follow the provider's supported compute
capabilities (9.x/10.x); they skip RTX 5090/SM120, where normal dispatch selects
another provider. CPU capture and contract checks remain applicable there.
The optional engine tests require a ShadowSpill version exporting the destination
and arbitrary-axis chunking APIs.

For changes to prepared kernels or these build contracts, also run the
maintained real-provider regression from the ShadowSpill checkout, with this
MLOps checkout and the pinned SonicMoE/Quack extras on its import path, on a
GPU supported by the SonicMoE provider:

```bash
python -m pytest -m slow tests/workloads/test_prepared_sonic.py
```

The two BF16 LoRA cases cover saved/recomputed training, nested chunks, shared
factor destinations, empty experts, cold saved profiling/execution and
checkpoint resume in another process. They compare independent CPU references,
check actual GEMM destination pointers and enforce physical memory admission.
Saved phases forbid provider compilation and tuning. This opt-in slow test is
separate from ShadowSpill's complete gates and distributed/full-model
qualification; see `tests/workloads/prepared_sonic/README.md` in that checkout
for the exact scope and optional ordinary build-cache setting.

For ordinary MLOps workload factories, use
`operator_preparations(dtype=torch.bfloat16)` from this integration module and
pass its returned tuple as `operator_preparation` in shared build options. The
description is CPU-only; the provider checks optional Sonic availability and
supported device capability when build opens it. Other devices keep ordinary
dispatch. The helper covers BF16 Sonic and separately records Hopper sparse
forward's backend and FlashMLA LSE convention, avoiding a first-call GPU probe
inside a measured task. It does not package the attention binaries or cuDNN
backward compiler. FP8/EP preparations and other external compilers need their
own qualified adapters. An explicitly
selected Sonic workload should use `sonicmoe_preparation()` instead.

Programs built with the earlier engine-owned Quack tuning records or v1 Sonic
preparations must be rebuilt and reprofiled. Saved v2 tasks own both choices and
binaries, including host-cache shortcuts; loader validation refuses missing or
incompatible state before execution can select another implementation.

The decomposed router also declares its own checkpoint default on the public
`torch.ops.mlops.lora_route.default` overload through the static boolean
`retain_for_backward` annotation. This imports no ShadowSpill code.
Both variants preserve backward-required expert IDs and weights by default;
explicit user retention rules can override it. Core has no MLOps router-name
check. The declaration changes neither routing arithmetic nor saved-Plan replay.
