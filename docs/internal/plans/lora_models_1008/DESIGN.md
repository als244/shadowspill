# LoRA across the supplied models

## Status and constraints

Work on Chicago, 2026-10-08. Baseline qualification passed on Chicago and
Tubingen before GPU tests. The separate LoRA head-loss operation is installed
and validated in MLOps; whole-model conversion remains in progress.

Accepted user decisions:

- Model definitions must remain readable and straightforward. Keep LoRA selection,
  freezing, conversion and checkpoint selection outside model forward methods.
- Expert blocks may have separate LoRA module implementations, following the
  existing QuackMoE/QuackMoELoRA and TEMoE/TEMoELoRA pairs.
- Embedding and output-head weights are frozen by default. Either can be fully
  trained by parameter name. The output head additionally supports opt-in LoRA;
  embeddings do not receive LoRA factors. The head therefore has three choices:
  frozen, LoRA, or full training.
- Earlier expert choices remain the starting point: independent factors per
  expert, joint gate/up factors, rank 32 and alpha 32, BF16 expert-factor compute
  and FP32 gradients. Do not change cross-expert sharing implicitly.
- Exact default module targeting is still being discussed. The proposed preset
  covers attention/mixer projections, dense MLPs and routed experts; shared
  experts remain explicitly selectable.

## Code audit

1. Both Llama implementations expose attention (`wq/wk/wv/wo`) and dense MLP
   (`w1/w3/w2`) projections as ordinary `nn.Linear`. Dense Qwen3.5 also exposes
   the full-attention projections and hybrid-mixer `w_qkvz/w_ba/w_out` this way.
2. OLMoE experts are packed 3D Parameters (`w13_experts`, `w2_experts`), not
   Linear children. MLOps calls `mlops.moe`; the PyTorch model computes an
   independent per-expert reference. Qwen MoE uses similarly packed `w13/w2`
   locally and QuackMoE in EP mode.
3. The MLOps models' logits and loss paths read `lm_head.weight` directly.
   Replacing only `lm_head.forward` would silently miss these callers. The
   head replacement must serve both logits and loss, including the common
   workload objective. A dedicated `mlops.lora_head_loss` operation preserves
   bounded logits and avoids materializing the dense low-rank weight update.
4. `mlops/providers/builtin/head.py::_run` currently allocates a complete
   `grad_head` and computes it even when the head is frozen.
   `providers/builtin/moe_composed.py::_prepare_backward` similarly computes
   expert and router gradients unconditionally. Frozen-weight execution must
   retain input gradients but omit unneeded parameter gradients inside these
   opaque regions; deleting their returned values later cannot remove work
   already performed inside a custom op.
5. ShadowSpill training already uses reachable parameter gradients and
   `requires_grad` when creating masters and optimizer state. This is the
   generic contract to retain. The ordinary PyTorch backend's optimizer
   parameter list still includes frozen parameters; audit state allocation and
   parameter-group behavior before changing that generic path.
6. Existing EP LoRAConfig imposes BF16 and multiples-of-16 ranks. These are
   implementation limits of those expert kernels, not universal LoRA limits.
   A generic dense implementation should accept arbitrary positive rank and
   validate supported dtypes independently.
7. Qwen can tie embeddings and head weights. Full-training selection acts on
   parameter identity, so selecting one alias trains the shared parameter.
   Initialization and checkpoint loading must preserve this tie.

## Recommended ownership and interface

Put reusable LoRA math, configuration, transforms and trainable-state inspection
in MLOps. Do not make ShadowSpill's planner or trainer recognize a LoRA mode.
The transformed object is an ordinary PyTorch module with frozen base weights
and explicit trainable low-rank factors.

Construction order:

1. Construct the usual model, including any caller-provided EP groups/buffers.
2. Apply the LoRA transformation before optimizer creation or ShadowSpill import.
3. Initialize/load the frozen base and initialize/restore factors, without
   resetting or copying already initialized base weights.
4. Pass the resulting model into ordinary training/planning APIs.

The transformation operates once, outside capture and execution. Dense modules
become `LoRALinear` modules with the same call signature. Expert modules become
specific LoRA variants with the same input/output structure, routing behavior,
metrics, parameter ownership and EP group semantics. The transform preserves
base parameter objects/names where possible rather than making another model
copy. It should reject unsupported explicit targets and duplicate application.

The core selector should accept module paths/globs (and a Python predicate when
needed). Model-family defaults belong in the supplied workload recipes, which
know which paths represent attention, MLP, router, head and shared experts. The
MLOps core must not import ShadowSpill workloads to guess those roles. Recipes
translate a short LoRA config into the same explicit transform.

Proposed low-level interface (not implemented):

```python
model = Llama3(config)
model = apply_lora(
    model,
    LoRAConfig(rank=32, alpha=32),
    targets=["blocks.*.attn.*", "blocks.*.mlp.*"],
    trainable_base=[],
)
```

`trainable_base=["lm_head.weight"]` additionally trains that complete parameter.
LoRA factors remain trainable. Everything else stays frozen. Existing optimizer
parameter groups specify different schedules/dtypes if desired. Workload config
can supply `lora: {rank: 32, alpha: 32}` and resolve its default targets; omitting
LoRA keeps ordinary full training. A startup report lists every selected module,
trainable parameter, dtype and parameter count.

The math stays as `linear(x, W) + scale * linear(linear(x, A), B)`.
Do not implement the production expert path by rebuilding a dense `W + B @ A`
for every expert on every call: that recreates full-sized temporaries and can
force custom operators to compute dense weight gradients. Use factor operations
inside dedicated expert LoRA execution instead. A merged-weight expression is
useful as an independent correctness oracle.

Keep default checkpoint resume behavior generic and correct first. Compact
LoRA-factor checkpoints should include explicitly trained base parameters and a
base-checkpoint identity. Never silently omit mutable buffers or data progress.
This should use a generic selected-state mechanism rather than LoRA branches in
the neutral runtime.

## Alternatives

- **Recommended:** ordinary dense replacements plus dedicated expert LoRA blocks.
  One config/selection surface; fused expert details stay in MLOps.
- **PEFT dependency:** useful ecosystem interoperability for ordinary layers.
  Packed expert parameters/custom operations still require integration. Its
  documented parameter targeting can materialize expert updates and has compiled
  execution caveats, so it is not an automatic solution to our fused kernels.
- **Weight parametrization everywhere:** concise reference prototype, but its
  materialized dense update loses important LoRA memory/work advantages here.
- **LoRA branches in every model:** rejected by the user's clarity requirement.

## External interface comparison, checked 2026-10-08

- Megatron Bridge defaults to QKV/output attention projections and MLP FC1/FC2.
  Module transformation occurs before distributed wrapping. Its grouped-expert
  factors default to sharing across local experts, with a per-expert option.
  [Official guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html)
- Unsloth's standard guidance targets Q/K/V/O and gate/up/down projections through
  `get_peft_model`; embedding/head targets are additional choices.
  [Official guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- Tinker's client defaults enable attention, MLP (including MoE), and output-head
  LoRA. Its three booleans are semantic targeting, not full-weight training.
  [Client API](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/)
  [LoRA config](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/loraconfig/)
- Tinker documents expert factors with the hidden-dimension-connected factor
  shared across experts and the other factor expert-specific. This differs from
  our existing independent factors per expert and is not proposed as an implicit
  change. [Parameter-count API](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/hyperparam_utils/get_lora_param_count/)
- [PEFT parameter-targeting documentation](https://huggingface.co/docs/peft/main/package_reference/lora#targeting-nnparameter-directly)

## Validation after qualification

- Dense and grouped-expert outputs/input gradients/factor gradients against an
  independent PyTorch merged-weight oracle; zero-initialized B and nonzero B.
- Frozen weights bitwise unchanged across updates, no frozen optimizer/master
  state or dense frozen-weight gradient allocations. Retain needed input grads.
- Both implementations of Llama3, Qwen3.5 and OLMoE; Qwen30B/Qwen35MoE recipes;
  existing Quack/TE expert variants where hardware supports them.
- Meta initialization, pretrained base import, tied parameters, factor-only/full
  checkpoint round trips and additional trainable-base selections.
- Eager, torch.compile, ShadowSpill save and recompute entrypoints with fresh
  stores. Inspect graphpair memory/object inventories, not only scalar losses.
- First single GPU on Chicago; DP and EP factor ownership/gradient reductions
  need subsequent multi-GPU validation. Do not claim EP based on dense tests.

## Output-head LoRA decision (2026-10-08)

The user explicitly selected a separate MLOps LoRA head-loss operation alongside
ordinary `head_loss`. Implemented call:

```python
loss = mlops.lora_head_loss(hidden, weight, lora_a, lora_b, targets,
                            scale=alpha / rank, chunk_size=chunk_size)
```

Shapes are `hidden[..., D]`, `weight[V, D]`, `lora_a[R, D]`,
`lora_b[V, R]`. Each row chunk evaluates `X W.T + scale * (X A.T) B.T`.
The chunk is consumed immediately by cross entropy. First-order input/factor
gradients are calculated in this pass and scaled by the incoming scalar
cotangent during backward, following the ordinary head-loss implementation.
Frozen base weights have no dense weight-gradient allocation or computation.
If the caller explicitly trains the base too (including a tied embedding),
its gradient is calculated. No full `B @ A` or full-batch logits are created.
The initial operation is deterministic and has no dropout argument.

The head module selects the operation; model forward methods should not contain
LoRA branches. Pure PyTorch reference heads keep independent ordinary autograd.
Validation covers nonzero factors as well as zero-initialized B, input/factor
gradients, ignored labels, chunk sizes, scalar cotangents, frozen state,
compilation, and saved-seed memory.
